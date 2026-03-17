use crate::error::LexError;
use crate::source_map::Span;
use crate::token::{keyword_lookup, Token, TokenKind};

/// Lexer that tokenizes a regulatory DSL source string.
pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    errors: Vec<LexError>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
            errors: Vec::new(),
        }
    }

    /// Tokenize the entire source, returning tokens and any lex errors.
    pub fn tokenize(mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        (tokens, self.errors)
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        if self.pos >= self.source.len() {
            return Token::eof(self.pos);
        }

        let start = self.pos;
        let ch = self.current_char();

        // String literal
        if ch == '"' {
            return self.lex_string();
        }

        // Date literal: #YYYY-MM-DD
        if ch == '#' {
            return self.lex_date();
        }

        // Number literal
        if ch.is_ascii_digit() {
            return self.lex_number();
        }

        // Identifier or keyword (allow underscore start)
        if ch.is_ascii_alphabetic() || ch == '_' {
            return self.lex_identifier_or_keyword();
        }

        // Unicode composition operators
        let remaining = &self.source[self.pos..];
        if remaining.starts_with('\u{2297}') {
            // ⊗
            let len = '\u{2297}'.len_utf8();
            self.pos += len;
            return Token::new(TokenKind::Conjunction, Span::new(start, self.pos));
        }
        if remaining.starts_with('\u{2295}') {
            // ⊕
            let len = '\u{2295}'.len_utf8();
            self.pos += len;
            return Token::new(TokenKind::Disjunction, Span::new(start, self.pos));
        }
        if remaining.starts_with('\u{25B7}') {
            // ▷
            let len = '\u{25B7}'.len_utf8();
            self.pos += len;
            return Token::new(TokenKind::OverrideOp, Span::new(start, self.pos));
        }
        if remaining.starts_with('\u{2298}') {
            // ⊘
            let len = '\u{2298}'.len_utf8();
            self.pos += len;
            return Token::new(TokenKind::ExceptionOp, Span::new(start, self.pos));
        }

        // Multi-char and single-char operators/punctuation
        self.lex_operator_or_punct(start)
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> Token {
        let saved_pos = self.pos;
        let saved_errors_len = self.errors.len();
        let tok = self.next_token();
        self.pos = saved_pos;
        self.errors.truncate(saved_errors_len);
        tok
    }

    // ─── Internal methods ───────────────────────────────────────

    fn current_char(&self) -> char {
        self.source[self.pos..].chars().next().unwrap_or('\0')
    }

    fn peek_char_at(&self, offset: usize) -> char {
        let pos = self.pos + offset;
        if pos < self.source.len() {
            self.source[pos..].chars().next().unwrap_or('\0')
        } else {
            '\0'
        }
    }

    fn advance(&mut self) {
        if self.pos < self.source.len() {
            let ch = self.current_char();
            self.pos += ch.len_utf8();
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            self.skip_whitespace();
            if !self.skip_comments() {
                break;
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.source.len() && self.current_char().is_whitespace() {
            self.advance();
        }
    }

    /// Attempt to skip a comment. Returns true if a comment was skipped.
    fn skip_comments(&mut self) -> bool {
        if self.pos + 1 >= self.source.len() {
            return false;
        }
        // Line comment //
        if self.bytes[self.pos] == b'/' && self.bytes[self.pos + 1] == b'/' {
            while self.pos < self.source.len() && self.current_char() != '\n' {
                self.advance();
            }
            return true;
        }
        // Block comment /* */
        if self.bytes[self.pos] == b'/' && self.bytes[self.pos + 1] == b'*' {
            let start = self.pos;
            self.pos += 2;
            let mut depth = 1;
            while self.pos + 1 < self.source.len() && depth > 0 {
                if self.bytes[self.pos] == b'/' && self.bytes[self.pos + 1] == b'*' {
                    depth += 1;
                    self.pos += 2;
                } else if self.bytes[self.pos] == b'*' && self.bytes[self.pos + 1] == b'/' {
                    depth -= 1;
                    self.pos += 2;
                } else {
                    self.advance();
                }
            }
            if depth > 0 {
                self.errors
                    .push(LexError::unterminated_comment(Span::new(start, self.pos)));
            }
            return true;
        }
        false
    }

    fn lex_identifier_or_keyword(&mut self) -> Token {
        let start = self.pos;
        while self.pos < self.source.len() {
            let ch = self.current_char();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        let word = &self.source[start..self.pos];
        let kind = keyword_lookup(word).unwrap_or_else(|| TokenKind::Ident(word.to_string()));
        Token::new(kind, Span::new(start, self.pos))
    }

    fn lex_number(&mut self) -> Token {
        let start = self.pos;
        let mut is_float = false;

        while self.pos < self.source.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        // Check for decimal point (but not .. range)
        if self.pos < self.source.len()
            && self.bytes[self.pos] == b'.'
            && self.pos + 1 < self.source.len()
            && self.bytes[self.pos + 1].is_ascii_digit()
        {
            is_float = true;
            self.pos += 1; // skip '.'
            while self.pos < self.source.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        let text = &self.source[start..self.pos];
        let span = Span::new(start, self.pos);

        if is_float {
            match text.parse::<f64>() {
                Ok(v) => Token::new(TokenKind::FloatLit(v), span),
                Err(_) => {
                    self.errors.push(LexError::invalid_number(span, text));
                    Token::new(TokenKind::FloatLit(0.0), span)
                }
            }
        } else {
            match text.parse::<i64>() {
                Ok(v) => Token::new(TokenKind::IntLit(v), span),
                Err(_) => {
                    self.errors.push(LexError::invalid_number(span, text));
                    Token::new(TokenKind::IntLit(0), span)
                }
            }
        }
    }

    fn lex_string(&mut self) -> Token {
        let start = self.pos;
        self.pos += 1; // skip opening "
        let mut value = String::new();

        while self.pos < self.source.len() {
            let ch = self.current_char();
            if ch == '"' {
                self.pos += 1;
                return Token::new(
                    TokenKind::StringLit(value),
                    Span::new(start, self.pos),
                );
            }
            if ch == '\\' && self.pos + 1 < self.source.len() {
                self.pos += 1;
                let esc = self.current_char();
                match esc {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    _ => {
                        value.push('\\');
                        value.push(esc);
                    }
                }
                self.advance();
            } else if ch == '\n' {
                // Newline in string without escape - error
                self.errors
                    .push(LexError::unterminated_string(Span::new(start, self.pos)));
                return Token::new(
                    TokenKind::StringLit(value),
                    Span::new(start, self.pos),
                );
            } else {
                value.push(ch);
                self.advance();
            }
        }

        self.errors
            .push(LexError::unterminated_string(Span::new(start, self.pos)));
        Token::new(TokenKind::StringLit(value), Span::new(start, self.pos))
    }

    fn lex_date(&mut self) -> Token {
        let start = self.pos;
        self.pos += 1; // skip '#'

        let date_start = self.pos;
        // Expect YYYY-MM-DD
        while self.pos < self.source.len() {
            let ch = self.current_char();
            if ch.is_ascii_digit() || ch == '-' {
                self.advance();
            } else {
                break;
            }
        }

        let date_str = &self.source[date_start..self.pos];
        let span = Span::new(start, self.pos);

        // Validate format
        if chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").is_ok() {
            Token::new(TokenKind::DateLit(date_str.to_string()), span)
        } else {
            self.errors.push(LexError::invalid_date(span, date_str));
            Token::new(TokenKind::DateLit(date_str.to_string()), span)
        }
    }

    fn lex_operator_or_punct(&mut self, start: usize) -> Token {
        let ch = self.current_char();
        let next = self.peek_char_at(ch.len_utf8());

        let (kind, len) = match (ch, next) {
            ('{', _) => (TokenKind::LBrace, 1),
            ('}', _) => (TokenKind::RBrace, 1),
            ('(', _) => (TokenKind::LParen, 1),
            (')', _) => (TokenKind::RParen, 1),
            ('[', _) => (TokenKind::LBracket, 1),
            (']', _) => (TokenKind::RBracket, 1),
            (';', _) => (TokenKind::Semicolon, 1),
            (':', _) => (TokenKind::Colon, 1),
            (',', _) => (TokenKind::Comma, 1),
            ('.', _) => (TokenKind::Dot, 1),
            ('+', _) => (TokenKind::Plus, 1),
            ('*', _) => (TokenKind::Star, 1),
            // ASCII composition operator alternatives
            ('&', '*') => (TokenKind::Conjunction, 2),
            ('|', '+') => (TokenKind::Disjunction, 2),
            ('\\', '-') => (TokenKind::ExceptionOp, 2),
            // Arrow and comparison
            ('-', '>') => (TokenKind::Arrow, 2),
            ('=', '>') => (TokenKind::FatArrow, 2),
            ('=', '=') => (TokenKind::EqEq, 2),
            ('!', '=') => (TokenKind::NotEq, 2),
            ('<', '=') => (TokenKind::LtEq, 2),
            ('>', '>') => (TokenKind::OverrideOp, 2),
            ('>', '=') => (TokenKind::GtEq, 2),
            ('=', _) => (TokenKind::Eq, 1),
            ('<', _) => (TokenKind::Lt, 1),
            ('>', _) => (TokenKind::Gt, 1),
            ('!', _) => (TokenKind::Not, 1),
            ('-', _) => (TokenKind::Minus, 1),
            ('/', _) => (TokenKind::Slash, 1),
            _ => {
                self.advance();
                self.errors
                    .push(LexError::unexpected_char(Span::new(start, self.pos), ch));
                return self.next_token();
            }
        };

        self.pos += len;
        Token::new(kind, Span::new(start, self.pos))
    }
}

/// Convenience function: lex a source string.
pub fn lex(source: &str) -> (Vec<Token>, Vec<LexError>) {
    Lexer::new(source).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_kinds(src: &str) -> Vec<TokenKind> {
        let (tokens, _) = lex(src);
        tokens.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty_input() {
        let kinds = lex_kinds("");
        assert_eq!(kinds, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_keywords() {
        let kinds = lex_kinds("jurisdiction obligation permission prohibition");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Jurisdiction,
                TokenKind::Obligation,
                TokenKind::Permission,
                TokenKind::Prohibition,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let kinds = lex_kinds("foo_bar baz123 _private");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident("foo_bar".into()),
                TokenKind::Ident("baz123".into()),
                TokenKind::Ident("_private".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_numbers() {
        let kinds = lex_kinds("42 3.14 0");
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLit(42),
                TokenKind::FloatLit(3.14),
                TokenKind::IntLit(0),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let kinds = lex_kinds(r#""hello world" "escaped\"quote""#);
        assert_eq!(
            kinds,
            vec![
                TokenKind::StringLit("hello world".into()),
                TokenKind::StringLit("escaped\"quote".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_date_literal() {
        let kinds = lex_kinds("#2024-08-01");
        assert_eq!(
            kinds,
            vec![TokenKind::DateLit("2024-08-01".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_punctuation() {
        let kinds = lex_kinds("{ } ( ) [ ] ; : , . -> =>");
        assert_eq!(
            kinds,
            vec![
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::Comma,
                TokenKind::Dot,
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_operators() {
        let kinds = lex_kinds("+ - * / = == != < > <= >=");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Eq,
                TokenKind::EqEq,
                TokenKind::NotEq,
                TokenKind::Lt,
                TokenKind::Gt,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_composition_operators_ascii() {
        let kinds = lex_kinds("&* |+ >> \\-");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Conjunction,
                TokenKind::Disjunction,
                TokenKind::OverrideOp,
                TokenKind::ExceptionOp,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_composition_operators_unicode() {
        let kinds = lex_kinds("⊗ ⊕ ▷ ⊘");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Conjunction,
                TokenKind::Disjunction,
                TokenKind::OverrideOp,
                TokenKind::ExceptionOp,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_line_comments() {
        let kinds = lex_kinds("foo // this is a comment\nbar");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident("foo".into()),
                TokenKind::Ident("bar".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_block_comments() {
        let kinds = lex_kinds("foo /* block\ncomment */ bar");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident("foo".into()),
                TokenKind::Ident("bar".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_nested_block_comments() {
        let kinds = lex_kinds("a /* outer /* inner */ still comment */ b");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::Ident("b".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_unterminated_string_error() {
        let (_, errors) = lex("\"unterminated");
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("unterminated"));
    }

    #[test]
    fn test_unexpected_char_recovery() {
        let (tokens, errors) = lex("foo @ bar");
        assert!(!errors.is_empty());
        // Should still produce tokens around the bad char
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert!(kinds.contains(&&TokenKind::Ident("foo".into())));
        assert!(kinds.contains(&&TokenKind::Ident("bar".into())));
    }

    #[test]
    fn test_risk_level_keyword() {
        let kinds = lex_kinds("risk_level");
        assert_eq!(kinds, vec![TokenKind::RiskLevel, TokenKind::Eof]);
    }

    #[test]
    fn test_complex_token_stream() {
        let src = r#"
            jurisdiction "EU" {
                obligation transparency {
                    risk_level: high;
                    formalizability: 2;
                    temporal: #2024-08-01 -> #2025-12-31;
                }
            }
        "#;
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "unexpected lex errors: {:?}", errors);
        // Just verify it produced a reasonable number of tokens
        assert!(tokens.len() > 10);
    }

    #[test]
    fn test_span_tracking() {
        let (tokens, _) = lex("ab cd");
        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[1].span, Span::new(3, 5));
    }
}
