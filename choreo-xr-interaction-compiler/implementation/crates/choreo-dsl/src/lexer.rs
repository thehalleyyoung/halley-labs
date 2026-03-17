//! Lexer for the Choreo DSL.
//!
//! Converts source text into a sequence of [`Token`]s, handling keywords,
//! numeric literals (with duration/distance/angle suffixes), string literals
//! with escape sequences, comments, and whitespace.

use crate::token::{DistanceUnit, DurationUnit, Token, TokenKind};
use choreo_types::{ChoreoError, Span};

/// Lexer state: scans source text character-by-character.
pub struct Lexer<'src> {
    source: &'src str,
    chars: Vec<char>,
    pos: usize,
    line: u32,
    column: u32,
    token_start: usize,
    token_start_line: u32,
    token_start_col: u32,
    errors: Vec<ChoreoError>,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source text.
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            token_start: 0,
            token_start_line: 1,
            token_start_col: 1,
            errors: Vec::new(),
        }
    }

    /// Lex the entire source and return all tokens (including EOF).
    pub fn lex(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    /// Lex the source, filtering out comments and newlines.
    pub fn lex_filtered(&mut self) -> Vec<Token> {
        self.lex()
            .into_iter()
            .filter(|t| {
                !matches!(t.kind, TokenKind::Comment(_) | TokenKind::Newline)
            })
            .collect()
    }

    /// Return accumulated lexer errors.
    pub fn errors(&self) -> &[ChoreoError] {
        &self.errors
    }

    /// Whether any errors were encountered.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    // -----------------------------------------------------------------------
    // Character scanning
    // -----------------------------------------------------------------------

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn advance_while(&mut self, pred: impl Fn(char) -> bool) {
        while let Some(ch) = self.peek() {
            if pred(ch) {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn mark_start(&mut self) {
        self.token_start = self.pos;
        self.token_start_line = self.line;
        self.token_start_col = self.column;
    }

    fn current_span(&self) -> Span {
        Span {
            start: self.token_start,
            end: self.pos,
            file: None,
        }
    }

    fn current_lexeme(&self) -> String {
        self.chars[self.token_start..self.pos].iter().collect()
    }

    fn make_token(&self, kind: TokenKind) -> Token {
        Token::new(kind, self.current_span(), self.current_lexeme())
    }

    fn error_token(&mut self, msg: &str) -> Token {
        self.errors.push(ChoreoError::Parse(format!(
            "{}:{}: {}",
            self.token_start_line, self.token_start_col, msg
        )));
        self.make_token(TokenKind::Identifier(format!("<error: {}>", msg)))
    }

    // -----------------------------------------------------------------------
    // Main scanning loop
    // -----------------------------------------------------------------------

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        self.mark_start();

        let ch = match self.advance() {
            Some(ch) => ch,
            None => return self.make_token(TokenKind::Eof),
        };

        match ch {
            '\n' => self.make_token(TokenKind::Newline),

            '(' => self.make_token(TokenKind::LParen),
            ')' => self.make_token(TokenKind::RParen),
            '{' => self.make_token(TokenKind::LBrace),
            '}' => self.make_token(TokenKind::RBrace),
            '[' => self.make_token(TokenKind::LBracket),
            ']' => self.make_token(TokenKind::RBracket),
            ',' => self.make_token(TokenKind::Comma),
            ';' => self.make_token(TokenKind::Semicolon),
            ':' => self.make_token(TokenKind::Colon),
            '@' => self.make_token(TokenKind::At),
            '#' => self.make_token(TokenKind::Hash),
            '+' => self.make_token(TokenKind::Plus),
            '*' => self.make_token(TokenKind::Star),
            '&' => self.make_token(TokenKind::Ampersand),
            '|' => self.make_token(TokenKind::Pipe),

            '.' => {
                if self.peek() == Some('.') {
                    self.advance();
                    self.make_token(TokenKind::DotDot)
                } else {
                    self.make_token(TokenKind::Dot)
                }
            }

            '-' => {
                if self.peek() == Some('>') {
                    self.advance();
                    self.make_token(TokenKind::Arrow)
                } else if self.peek().map_or(false, |c| c.is_ascii_digit()) {
                    self.scan_number()
                } else {
                    self.make_token(TokenKind::Minus)
                }
            }

            '=' => {
                if self.peek() == Some('>') {
                    self.advance();
                    self.make_token(TokenKind::FatArrow)
                } else {
                    self.make_token(TokenKind::Eq)
                }
            }

            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.make_token(TokenKind::Le)
                } else {
                    self.make_token(TokenKind::Lt)
                }
            }

            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.make_token(TokenKind::Ge)
                } else {
                    self.make_token(TokenKind::Gt)
                }
            }

            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.make_token(TokenKind::Ne)
                } else {
                    self.make_token(TokenKind::Not)
                }
            }

            '/' => {
                if self.peek() == Some('/') {
                    self.scan_line_comment()
                } else if self.peek() == Some('*') {
                    self.scan_block_comment()
                } else {
                    self.make_token(TokenKind::Slash)
                }
            }

            '"' => self.scan_string(),

            c if c.is_ascii_digit() => self.scan_number(),

            c if is_ident_start(c) => self.scan_identifier(),

            other => self.error_token(&format!("unexpected character '{}'", other)),
        }
    }

    // -----------------------------------------------------------------------
    // Whitespace & comments
    // -----------------------------------------------------------------------

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            match ch {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn scan_line_comment(&mut self) -> Token {
        self.advance(); // consume second '/'
        let start = self.pos;
        self.advance_while(|c| c != '\n');
        let text: String = self.chars[start..self.pos].iter().collect();
        self.make_token(TokenKind::Comment(text.trim().to_string()))
    }

    fn scan_block_comment(&mut self) -> Token {
        self.advance(); // consume '*'
        let start = self.pos;
        let mut depth = 1u32;
        while !self.at_end() && depth > 0 {
            if self.peek() == Some('/') && self.peek_next() == Some('*') {
                self.advance();
                self.advance();
                depth += 1;
            } else if self.peek() == Some('*') && self.peek_next() == Some('/') {
                self.advance();
                self.advance();
                depth -= 1;
            } else {
                self.advance();
            }
        }
        if depth > 0 {
            return self.error_token("unterminated block comment");
        }
        let end = if self.pos >= 2 { self.pos - 2 } else { self.pos };
        let text: String = self.chars[start..end].iter().collect();
        self.make_token(TokenKind::Comment(text.trim().to_string()))
    }

    // -----------------------------------------------------------------------
    // String literals
    // -----------------------------------------------------------------------

    fn scan_string(&mut self) -> Token {
        let mut value = String::new();
        loop {
            match self.advance() {
                None => return self.error_token("unterminated string literal"),
                Some('"') => break,
                Some('\\') => match self.advance() {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('r') => value.push('\r'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some('0') => value.push('\0'),
                    Some('u') => {
                        if self.peek() == Some('{') {
                            self.advance();
                            let hex_start = self.pos;
                            self.advance_while(|c| c.is_ascii_hexdigit());
                            let hex_str: String =
                                self.chars[hex_start..self.pos].iter().collect();
                            if self.peek() == Some('}') {
                                self.advance();
                                if let Ok(code) = u32::from_str_radix(&hex_str, 16) {
                                    if let Some(ch) = char::from_u32(code) {
                                        value.push(ch);
                                    } else {
                                        return self.error_token(&format!(
                                            "invalid unicode scalar value: U+{:X}",
                                            code
                                        ));
                                    }
                                } else {
                                    return self.error_token("invalid unicode escape");
                                }
                            } else {
                                return self.error_token("expected '}' in unicode escape");
                            }
                        } else {
                            return self.error_token("expected '{' in unicode escape");
                        }
                    }
                    Some(c) => {
                        return self.error_token(&format!("invalid escape sequence '\\{}'", c));
                    }
                    None => return self.error_token("unterminated escape sequence"),
                },
                Some(c) => value.push(c),
            }
        }
        self.make_token(TokenKind::StringLiteral(value))
    }

    // -----------------------------------------------------------------------
    // Numeric literals (int, float, with optional suffix)
    // -----------------------------------------------------------------------

    fn scan_number(&mut self) -> Token {
        let negative = self.chars.get(self.token_start).copied() == Some('-');
        self.advance_while(|c| c.is_ascii_digit());

        let is_float = self.peek() == Some('.')
            && self.peek_next().map_or(false, |c| c.is_ascii_digit());

        if is_float {
            self.advance(); // consume '.'
            self.advance_while(|c| c.is_ascii_digit());
        }

        // Check for scientific notation
        if self.peek() == Some('e') || self.peek() == Some('E') {
            self.advance();
            if self.peek() == Some('+') || self.peek() == Some('-') {
                self.advance();
            }
            self.advance_while(|c| c.is_ascii_digit());
            return self.finish_number_with_suffix(true, negative);
        }

        self.finish_number_with_suffix(is_float, negative)
    }

    fn finish_number_with_suffix(&mut self, is_float: bool, _negative: bool) -> Token {
        let num_end = self.pos;
        let num_str: String = self.chars[self.token_start..num_end].iter().collect();

        // Peek at suffix
        let suffix_start = self.pos;
        self.advance_while(|c| c.is_ascii_alphabetic() || c == '_');
        let suffix: String = self.chars[suffix_start..self.pos].iter().collect();

        if suffix.is_empty() {
            if is_float {
                match num_str.parse::<f64>() {
                    Ok(v) => self.make_token(TokenKind::FloatLiteral(v)),
                    Err(_) => self.error_token(&format!("invalid float literal: {}", num_str)),
                }
            } else {
                match num_str.parse::<i64>() {
                    Ok(v) => self.make_token(TokenKind::IntLiteral(v)),
                    Err(_) => self.error_token(&format!("invalid integer literal: {}", num_str)),
                }
            }
        } else {
            let value = match num_str.parse::<f64>() {
                Ok(v) => v,
                Err(_) => {
                    return self.error_token(&format!("invalid numeric literal: {}", num_str));
                }
            };
            match suffix.as_str() {
                "ms" => self.make_token(TokenKind::DurationLiteral(value, DurationUnit::Ms)),
                "s" => self.make_token(TokenKind::DurationLiteral(value, DurationUnit::S)),
                "min" => self.make_token(TokenKind::DurationLiteral(value, DurationUnit::Min)),
                "mm" => self.make_token(TokenKind::DistanceLiteral(value, DistanceUnit::Mm)),
                "cm" => self.make_token(TokenKind::DistanceLiteral(value, DistanceUnit::Cm)),
                "m" => self.make_token(TokenKind::DistanceLiteral(value, DistanceUnit::M)),
                "deg" => self.make_token(TokenKind::AngleLiteral(value)),
                _ => self.error_token(&format!("unknown numeric suffix '{}'", suffix)),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Identifiers & keywords
    // -----------------------------------------------------------------------

    fn scan_identifier(&mut self) -> Token {
        self.advance_while(|c| is_ident_continue(c));
        let text: String = self.chars[self.token_start..self.pos].iter().collect();

        if let Some(kind) = TokenKind::from_keyword(&text) {
            self.make_token(kind)
        } else {
            self.make_token(TokenKind::Identifier(text))
        }
    }
}

// ---------------------------------------------------------------------------
// Character classification helpers
// ---------------------------------------------------------------------------

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Lex source text into tokens, filtering out comments and newlines.
pub fn lex(source: &str) -> Result<Vec<Token>, ChoreoError> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.lex_filtered();
    if lexer.has_errors() {
        let msgs: Vec<String> = lexer
            .errors()
            .iter()
            .map(|e| format!("{}", e))
            .collect();
        Err(ChoreoError::Parse(msgs.join("; ")))
    } else {
        Ok(tokens)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_kinds(src: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(src);
        lexer
            .lex_filtered()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn lex_all(src: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(src);
        lexer.lex_filtered()
    }

    #[test]
    fn test_empty_input() {
        let kinds = lex_kinds("");
        assert_eq!(kinds, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_punctuation() {
        let kinds = lex_kinds("( ) { } [ ] , ; : . + * & | @ #");
        assert_eq!(
            kinds,
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::Dot,
                TokenKind::Plus,
                TokenKind::Star,
                TokenKind::Ampersand,
                TokenKind::Pipe,
                TokenKind::At,
                TokenKind::Hash,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_multi_char_operators() {
        let kinds = lex_kinds("-> => <= >= != ..");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::Le,
                TokenKind::Ge,
                TokenKind::Ne,
                TokenKind::DotDot,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        let kinds = lex_kinds("< > = !");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Lt,
                TokenKind::Gt,
                TokenKind::Eq,
                TokenKind::Not,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_keywords() {
        let kinds = lex_kinds("region interaction scene entity zone");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Region,
                TokenKind::Interaction,
                TokenKind::Scene,
                TokenKind::Entity,
                TokenKind::Zone,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_gesture_keywords() {
        let kinds = lex_kinds("gaze reach grab release proximity inside contains touch");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Gaze,
                TokenKind::Reach,
                TokenKind::Grab,
                TokenKind::Release,
                TokenKind::Proximity,
                TokenKind::Inside,
                TokenKind::Contains,
                TokenKind::Touch,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_choreography_keywords() {
        let kinds = lex_kinds("par seq choice loop when then");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Par,
                TokenKind::Seq,
                TokenKind::Choice,
                TokenKind::Loop,
                TokenKind::When,
                TokenKind::Then,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let kinds = lex_kinds("my_region foo123 _bar Region2");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier("my_region".into()),
                TokenKind::Identifier("foo123".into()),
                TokenKind::Identifier("_bar".into()),
                TokenKind::Identifier("Region2".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_integer_literals() {
        let kinds = lex_kinds("0 42 12345");
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLiteral(0),
                TokenKind::IntLiteral(42),
                TokenKind::IntLiteral(12345),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_float_literals() {
        let tokens = lex_all("3.14 0.5 100.0");
        assert_eq!(tokens.len(), 4); // 3 floats + EOF
        match &tokens[0].kind {
            TokenKind::FloatLiteral(v) => assert!((*v - 3.14).abs() < 1e-9),
            other => panic!("expected float, got {:?}", other),
        }
        match &tokens[1].kind {
            TokenKind::FloatLiteral(v) => assert!((*v - 0.5).abs() < 1e-9),
            other => panic!("expected float, got {:?}", other),
        }
    }

    #[test]
    fn test_negative_number() {
        let kinds = lex_kinds("-42");
        assert_eq!(kinds, vec![TokenKind::IntLiteral(-42), TokenKind::Eof]);
    }

    #[test]
    fn test_duration_literals() {
        let kinds = lex_kinds("500ms 2s 1.5min");
        assert_eq!(
            kinds,
            vec![
                TokenKind::DurationLiteral(500.0, DurationUnit::Ms),
                TokenKind::DurationLiteral(2.0, DurationUnit::S),
                TokenKind::DurationLiteral(1.5, DurationUnit::Min),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_distance_literals() {
        let kinds = lex_kinds("30cm 1.5m 250mm");
        assert_eq!(
            kinds,
            vec![
                TokenKind::DistanceLiteral(30.0, DistanceUnit::Cm),
                TokenKind::DistanceLiteral(1.5, DistanceUnit::M),
                TokenKind::DistanceLiteral(250.0, DistanceUnit::Mm),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_angle_literal() {
        let kinds = lex_kinds("45deg 90deg");
        assert_eq!(
            kinds,
            vec![
                TokenKind::AngleLiteral(45.0),
                TokenKind::AngleLiteral(90.0),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let kinds = lex_kinds(r#""hello world""#);
        assert_eq!(
            kinds,
            vec![
                TokenKind::StringLiteral("hello world".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_string_escape_sequences() {
        let kinds = lex_kinds(r#""line\nnewline\ttab\\slash\"quote""#);
        assert_eq!(
            kinds,
            vec![
                TokenKind::StringLiteral("line\nnewline\ttab\\slash\"quote".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_string_unicode_escape() {
        let kinds = lex_kinds(r#""\u{0041}""#);
        assert_eq!(
            kinds,
            vec![TokenKind::StringLiteral("A".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_line_comment() {
        let mut lexer = Lexer::new("foo // this is a comment\nbar");
        let tokens = lexer.lex();
        let no_ws: Vec<_> = tokens
            .iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline))
            .collect();
        assert!(no_ws.iter().any(|t| matches!(&t.kind, TokenKind::Comment(s) if s == "this is a comment")));
    }

    #[test]
    fn test_block_comment() {
        let mut lexer = Lexer::new("foo /* block\ncomment */ bar");
        let tokens = lexer.lex_filtered();
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert!(kinds.contains(&&TokenKind::Identifier("foo".into())));
        assert!(kinds.contains(&&TokenKind::Identifier("bar".into())));
    }

    #[test]
    fn test_nested_block_comment() {
        let mut lexer = Lexer::new("a /* outer /* inner */ still outer */ b");
        let tokens = lexer.lex_filtered();
        let idents: Vec<_> = tokens
            .iter()
            .filter_map(|t| t.as_identifier().map(String::from))
            .collect();
        assert_eq!(idents, vec!["a", "b"]);
    }

    #[test]
    fn test_boolean_literals() {
        let kinds = lex_kinds("true false");
        assert_eq!(kinds, vec![TokenKind::True, TokenKind::False, TokenKind::Eof]);
    }

    #[test]
    fn test_geometry_keywords() {
        let kinds = lex_kinds("box sphere capsule cylinder convex_hull union intersection difference transform");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Box_,
                TokenKind::Sphere_,
                TokenKind::Capsule_,
                TokenKind::Cylinder_,
                TokenKind::ConvexHull,
                TokenKind::Union,
                TokenKind::Intersection,
                TokenKind::Difference,
                TokenKind::Transform,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_span_tracking() {
        let tokens = lex_all("ab cd");
        assert_eq!(tokens[0].span.start, 0);
        assert_eq!(tokens[0].span.end, 2);
        assert_eq!(tokens[1].span.start, 3);
        assert_eq!(tokens[1].span.end, 5);
    }

    #[test]
    fn test_complex_expression() {
        let kinds = lex_kinds("gaze(user, target, 15deg) and proximity(a, b, 1.5m)");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Gaze,
                TokenKind::LParen,
                TokenKind::Identifier("user".into()),
                TokenKind::Comma,
                TokenKind::Identifier("target".into()),
                TokenKind::Comma,
                TokenKind::AngleLiteral(15.0),
                TokenKind::RParen,
                TokenKind::And,
                TokenKind::Proximity,
                TokenKind::LParen,
                TokenKind::Identifier("a".into()),
                TokenKind::Comma,
                TokenKind::Identifier("b".into()),
                TokenKind::Comma,
                TokenKind::DistanceLiteral(1.5, DistanceUnit::M),
                TokenKind::RParen,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_interaction_declaration() {
        let src = r#"
            interaction grab_object(user, obj) {
                when gaze(user, obj) and reach(user, obj) then {
                    activate(obj);
                }
            }
        "#;
        let tokens = lex_all(src);
        assert!(tokens.len() > 10);
        assert_eq!(tokens[0].kind, TokenKind::Interaction);
    }

    #[test]
    fn test_unterminated_string_error() {
        let mut lexer = Lexer::new("\"hello");
        let _tokens = lexer.lex();
        assert!(lexer.has_errors());
    }

    #[test]
    fn test_unterminated_block_comment_error() {
        let mut lexer = Lexer::new("/* never closed");
        let _tokens = lexer.lex();
        assert!(lexer.has_errors());
    }

    #[test]
    fn test_unknown_suffix_error() {
        let mut lexer = Lexer::new("42xyz");
        let _tokens = lexer.lex();
        assert!(lexer.has_errors());
    }

    #[test]
    fn test_scientific_notation() {
        let kinds = lex_kinds("1e3 2.5E-4");
        match &kinds[0] {
            TokenKind::FloatLiteral(v) => assert!((*v - 1000.0).abs() < 1e-3),
            other => panic!("expected float, got {:?}", other),
        }
        match &kinds[1] {
            TokenKind::FloatLiteral(v) => assert!((*v - 0.00025).abs() < 1e-9),
            other => panic!("expected float, got {:?}", other),
        }
    }

    #[test]
    fn test_lex_convenience_function() {
        let tokens = lex("region my_zone { }").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Region);
        assert_eq!(tokens[1].kind, TokenKind::Identifier("my_zone".into()));
        assert_eq!(tokens[2].kind, TokenKind::LBrace);
        assert_eq!(tokens[3].kind, TokenKind::RBrace);
    }

    #[test]
    fn test_action_keywords() {
        let kinds = lex_kinds("activate deactivate emit spawn destroy set_timer cancel_timer");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Activate,
                TokenKind::Deactivate,
                TokenKind::Emit,
                TokenKind::Spawn,
                TokenKind::Destroy,
                TokenKind::SetTimer,
                TokenKind::CancelTimer,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_temporal_keywords() {
        let kinds = lex_kinds("timeout within after");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Timeout,
                TokenKind::Within,
                TokenKind::After,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_slash_is_not_comment() {
        let kinds = lex_kinds("a / b");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier("a".into()),
                TokenKind::Slash,
                TokenKind::Identifier("b".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_minus_as_operator() {
        let kinds = lex_kinds("a - b");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Identifier("a".into()),
                TokenKind::Minus,
                TokenKind::Identifier("b".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_import_keyword() {
        let kinds = lex_kinds("import");
        assert_eq!(kinds, vec![TokenKind::Import, TokenKind::Eof]);
    }

    #[test]
    fn test_newlines_in_unfiltered() {
        let mut lexer = Lexer::new("a\nb");
        let tokens = lexer.lex();
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Newline));
    }

    #[test]
    fn test_many_tokens_stress() {
        let src = (0..200).map(|i| format!("x{}", i)).collect::<Vec<_>>().join(" ");
        let tokens = lex_all(&src);
        assert_eq!(tokens.len(), 201); // 200 identifiers + EOF
    }
}
