//! Lexer / tokenizer for the SoniType DSL.
//!
//! Converts a source string into a sequence of [`Token`]s. Handles whitespace,
//! comments (line `//` and block `/* */`), string escaping, number literals
//! (integers, floats, scientific notation), keyword recognition, and multi-char
//! operator tokenization.  On encountering an invalid character the lexer
//! records an error and skips to the next valid token.

use crate::token::{LexError, Position, Span, Token, TokenKind, lookup_keyword};

// ─── Lexer ───────────────────────────────────────────────────────────────────

/// Lexer state: walks through the source string producing tokens.
pub struct Lexer<'src> {
    source: &'src str,
    chars: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    tokens: Vec<Token>,
    errors: Vec<LexError>,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            tokens: Vec::new(),
            errors: Vec::new(),
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn current_position(&self) -> Position {
        Position::new(self.line, self.column, self.pos)
    }

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

    fn emit(&mut self, kind: TokenKind, start: Position) {
        let end = self.current_position();
        self.tokens.push(Token::new(kind, Span::new(start, end)));
    }

    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(LexError::new(message, span));
    }

    // ── Skip whitespace and comments ─────────────────────────────────────────

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Whitespace
            self.advance_while(|c| c.is_ascii_whitespace());

            // Line comment
            if self.peek() == Some('/') && self.peek_next() == Some('/') {
                self.advance(); // /
                self.advance(); // /
                self.advance_while(|c| c != '\n');
                continue;
            }

            // Block comment
            if self.peek() == Some('/') && self.peek_next() == Some('*') {
                let start = self.current_position();
                self.advance(); // /
                self.advance(); // *
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
                        None => {
                            let end = self.current_position();
                            self.error("unterminated block comment", Span::new(start, end));
                            return;
                        }
                        _ => {}
                    }
                }
                continue;
            }

            break;
        }
    }

    // ── Number literal ───────────────────────────────────────────────────────

    fn lex_number(&mut self) {
        let start = self.current_position();
        let num_start = self.pos;
        let mut is_float = false;

        // Integer part
        self.advance_while(|c| c.is_ascii_digit() || c == '_');

        // Fractional part
        if self.peek() == Some('.') && self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
            self.advance(); // .
            self.advance_while(|c| c.is_ascii_digit() || c == '_');
        }

        // Exponent
        if matches!(self.peek(), Some('e') | Some('E')) {
            is_float = true;
            self.advance(); // e/E
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.advance();
            }
            self.advance_while(|c| c.is_ascii_digit());
        }

        let text: String = self.chars[num_start..self.pos]
            .iter()
            .filter(|c| **c != '_')
            .collect();

        if is_float {
            match text.parse::<f64>() {
                Ok(v) => self.emit(TokenKind::FloatLit(v), start),
                Err(_) => {
                    let end = self.current_position();
                    self.error(format!("invalid float literal: {text}"), Span::new(start, end));
                }
            }
        } else {
            match text.parse::<i64>() {
                Ok(v) => self.emit(TokenKind::IntLit(v), start),
                Err(_) => {
                    let end = self.current_position();
                    self.error(format!("invalid integer literal: {text}"), Span::new(start, end));
                }
            }
        }
    }

    // ── String literal ───────────────────────────────────────────────────────

    fn lex_string(&mut self) {
        let start = self.current_position();
        self.advance(); // opening "
        let mut value = String::new();

        loop {
            match self.advance() {
                Some('"') => {
                    self.emit(TokenKind::StringLit(value), start);
                    return;
                }
                Some('\\') => match self.advance() {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('r') => value.push('\r'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some('0') => value.push('\0'),
                    Some(c) => {
                        let end = self.current_position();
                        self.error(
                            format!("unknown escape sequence: \\{c}"),
                            Span::new(start, end),
                        );
                        value.push(c);
                    }
                    None => {
                        let end = self.current_position();
                        self.error("unterminated string literal", Span::new(start, end));
                        return;
                    }
                },
                Some('\n') => {
                    let end = self.current_position();
                    self.error("unterminated string literal (newline)", Span::new(start, end));
                    return;
                }
                Some(c) => value.push(c),
                None => {
                    let end = self.current_position();
                    self.error("unterminated string literal", Span::new(start, end));
                    return;
                }
            }
        }
    }

    // ── Identifier / keyword ─────────────────────────────────────────────────

    fn lex_identifier(&mut self) {
        let start = self.current_position();
        let id_start = self.pos;
        self.advance_while(|c| c.is_alphanumeric() || c == '_');
        let text: String = self.chars[id_start..self.pos].iter().collect();

        let kind = lookup_keyword(&text).unwrap_or(TokenKind::Ident(text));
        self.emit(kind, start);
    }

    // ── Operator / delimiter ─────────────────────────────────────────────────

    fn lex_operator_or_delimiter(&mut self) {
        let start = self.current_position();
        let ch = self.advance().unwrap();

        let kind = match ch {
            '+' => TokenKind::Plus,
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '@' => TokenKind::At,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            ':' => TokenKind::Colon,
            '-' => {
                if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::EqEq
                } else {
                    TokenKind::Eq
                }
            }
            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                }
            }
            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            '|' => {
                if self.peek() == Some('|') {
                    self.advance();
                    TokenKind::PipePipe
                } else if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::PipeGt
                } else {
                    let end = self.current_position();
                    self.error("unexpected character '|'; did you mean '||' or '|>'?", Span::new(start, end));
                    return;
                }
            }
            '&' => {
                if self.peek() == Some('&') {
                    self.advance();
                    TokenKind::AmpAmp
                } else {
                    let end = self.current_position();
                    self.error("unexpected character '&'; did you mean '&&'?", Span::new(start, end));
                    return;
                }
            }
            '.' => {
                if self.peek() == Some('.') {
                    self.advance();
                    TokenKind::DotDot
                } else {
                    TokenKind::Dot
                }
            }
            _ => {
                let end = self.current_position();
                self.error(format!("unexpected character: '{ch}'"), Span::new(start, end));
                return;
            }
        };

        self.emit(kind, start);
    }

    // ── Main tokenize loop ───────────────────────────────────────────────────

    fn tokenize(&mut self) {
        loop {
            self.skip_whitespace_and_comments();

            match self.peek() {
                None => {
                    let pos = self.current_position();
                    self.tokens.push(Token::eof(pos.offset, pos.line, pos.column));
                    break;
                }
                Some(ch) => {
                    if ch.is_ascii_digit() {
                        self.lex_number();
                    } else if ch == '"' {
                        self.lex_string();
                    } else if ch.is_alphabetic() || ch == '_' {
                        self.lex_identifier();
                    } else {
                        self.lex_operator_or_delimiter();
                    }
                }
            }
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Lex a source string into tokens, collecting any errors encountered.
pub fn lex(source: &str) -> Result<Vec<Token>, Vec<LexError>> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    if lexer.errors.is_empty() {
        Ok(lexer.tokens)
    } else {
        Err(lexer.errors)
    }
}

/// Lex a source string, returning tokens and errors together.
pub fn lex_with_errors(source: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    (lexer.tokens, lexer.errors)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenKind::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty_source() {
        let tokens = lex("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, Eof);
    }

    #[test]
    fn test_whitespace_only() {
        let tokens = lex("   \n\t\r\n  ").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, Eof);
    }

    #[test]
    fn test_line_comment() {
        let k = kinds("42 // this is a comment\n43");
        assert_eq!(k, vec![IntLit(42), IntLit(43), Eof]);
    }

    #[test]
    fn test_block_comment() {
        let k = kinds("1 /* block */ 2");
        assert_eq!(k, vec![IntLit(1), IntLit(2), Eof]);
    }

    #[test]
    fn test_nested_block_comment() {
        let k = kinds("1 /* outer /* inner */ still comment */ 2");
        assert_eq!(k, vec![IntLit(1), IntLit(2), Eof]);
    }

    #[test]
    fn test_integer_literals() {
        let k = kinds("0 42 1_000");
        assert_eq!(k, vec![IntLit(0), IntLit(42), IntLit(1000), Eof]);
    }

    #[test]
    fn test_float_literals() {
        let k = kinds("3.14 0.5 1.0e10 2.5E-3");
        assert_eq!(
            k,
            vec![FloatLit(3.14), FloatLit(0.5), FloatLit(1.0e10), FloatLit(2.5e-3), Eof]
        );
    }

    #[test]
    fn test_string_literal() {
        let k = kinds(r#""hello" "world""#);
        assert_eq!(
            k,
            vec![StringLit("hello".into()), StringLit("world".into()), Eof]
        );
    }

    #[test]
    fn test_string_escapes() {
        let k = kinds(r#""a\nb\tc\\d\"e""#);
        assert_eq!(k, vec![StringLit("a\nb\tc\\d\"e".into()), Eof]);
    }

    #[test]
    fn test_keywords() {
        let k = kinds("stream mapping compose data let in with where if then else import export spec");
        assert_eq!(
            k,
            vec![
                Stream, Mapping, Compose, Data, Let, In, With, Where,
                If, Then, Else, Import, Export, Spec, Eof,
            ]
        );
    }

    #[test]
    fn test_boolean_keywords() {
        let k = kinds("true false");
        assert_eq!(k, vec![True, False, Eof]);
    }

    #[test]
    fn test_type_keywords() {
        let k = kinds("Stream Mapping MultiStream Data SonificationSpec Pitch Timbre Pan Amplitude Duration Float Int Bool String");
        assert_eq!(
            k,
            vec![
                TyStream, TyMapping, TyMultiStream, TyData, TySonificationSpec,
                TyPitch, TyTimbre, TyPan, TyAmplitude, TyDuration,
                TyFloat, TyInt, TyBool, TyString, Eof,
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let k = kinds("foo _bar baz123 my_stream");
        assert_eq!(
            k,
            vec![
                Ident("foo".into()),
                Ident("_bar".into()),
                Ident("baz123".into()),
                Ident("my_stream".into()),
                Eof,
            ]
        );
    }

    #[test]
    fn test_arithmetic_operators() {
        let k = kinds("+ - * /");
        assert_eq!(k, vec![Plus, Minus, Star, Slash, Eof]);
    }

    #[test]
    fn test_comparison_operators() {
        let k = kinds("== != < > <= >=");
        assert_eq!(k, vec![EqEq, BangEq, Lt, Gt, LtEq, GtEq, Eof]);
    }

    #[test]
    fn test_logical_operators() {
        let k = kinds("|| && !");
        assert_eq!(k, vec![PipePipe, AmpAmp, Bang, Eof]);
    }

    #[test]
    fn test_special_operators() {
        let k = kinds(".. |> @ = . ->");
        assert_eq!(k, vec![DotDot, PipeGt, At, Eq, Dot, Arrow, Eof]);
    }

    #[test]
    fn test_delimiters() {
        let k = kinds("{ } ( ) [ ] , ; :");
        assert_eq!(
            k,
            vec![LBrace, RBrace, LParen, RParen, LBracket, RBracket, Comma, Semicolon, Colon, Eof]
        );
    }

    #[test]
    fn test_stream_literal() {
        let k = kinds(r#"stream { freq: 440.0, timbre: "sine", pan: 0.0 }"#);
        assert_eq!(
            k,
            vec![
                Stream, LBrace,
                Ident("freq".into()), Colon, FloatLit(440.0), Comma,
                Ident("timbre".into()), Colon, StringLit("sine".into()), Comma,
                Ident("pan".into()), Colon, FloatLit(0.0),
                RBrace, Eof,
            ]
        );
    }

    #[test]
    fn test_mapping_declaration() {
        let k = kinds("mapping m = data.temperature -> pitch(200..800)");
        assert_eq!(
            k,
            vec![
                Mapping, Ident("m".into()), Eq,
                Data, Dot, Ident("temperature".into()),
                Arrow, Ident("pitch".into()), LParen, IntLit(200), DotDot, IntLit(800), RParen,
                Eof,
            ]
        );
    }

    #[test]
    fn test_compose_expression() {
        let k = kinds("compose { s1 || s2 || s3 }");
        assert_eq!(
            k,
            vec![
                Compose, LBrace,
                Ident("s1".into()), PipePipe,
                Ident("s2".into()), PipePipe,
                Ident("s3".into()),
                RBrace, Eof,
            ]
        );
    }

    #[test]
    fn test_position_tracking() {
        let tokens = lex("let x = 42").unwrap();
        assert_eq!(tokens[0].span.start.line, 1);
        assert_eq!(tokens[0].span.start.column, 1);
        // "x" starts at column 5
        assert_eq!(tokens[1].span.start.column, 5);
    }

    #[test]
    fn test_multiline_position_tracking() {
        let tokens = lex("let\n  x\n  = 42").unwrap();
        // "x" is on line 2
        assert_eq!(tokens[1].span.start.line, 2);
        // "42" is on line 3
        assert_eq!(tokens[3].span.start.line, 3);
    }

    #[test]
    fn test_error_recovery_unexpected_char() {
        let (tokens, errors) = lex_with_errors("42 # 43");
        assert!(!errors.is_empty());
        // Should still find the integers
        let int_tokens: Vec<_> = tokens.iter().filter(|t| matches!(t.kind, IntLit(_))).collect();
        assert_eq!(int_tokens.len(), 2);
    }

    #[test]
    fn test_unterminated_string_error() {
        let result = lex(r#""hello"#);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("unterminated"));
    }

    #[test]
    fn test_scientific_notation_negative_exponent() {
        let k = kinds("1.5e-3");
        assert_eq!(k, vec![FloatLit(1.5e-3), Eof]);
    }

    #[test]
    fn test_scientific_notation_positive_exponent() {
        let k = kinds("2e+5");
        assert_eq!(k, vec![FloatLit(2e5), Eof]);
    }

    #[test]
    fn test_pipe_operator() {
        let k = kinds("data |> filter |> map");
        assert_eq!(
            k,
            vec![
                Data, PipeGt,
                Ident("filter".into()), PipeGt,
                Ident("map".into()),
                Eof,
            ]
        );
    }

    #[test]
    fn test_complex_expression() {
        let k = kinds("if x > 0 then x * 2 else 0 - x");
        assert_eq!(
            k,
            vec![
                If, Ident("x".into()), Gt, IntLit(0),
                Then, Ident("x".into()), Star, IntLit(2),
                Else, IntLit(0), Minus, Ident("x".into()),
                Eof,
            ]
        );
    }
}
