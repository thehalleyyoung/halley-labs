//! S-expression parser for SMT-LIB2 solver responses.
//!
//! Handles atoms (symbols, numerals, strings), nested lists, and line
//! comments. Used to parse `(model …)`, `(values …)`, unsat cores, and
//! error messages emitted by Z3 / CVC5.

use std::fmt;

// ───────────────────────────── SExp AST ──────────────────────────────────

/// A single s-expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExp {
    /// An atomic token (symbol, numeral, keyword, quoted string content).
    Atom(String),
    /// A parenthesised list of s-expressions.
    List(Vec<SExp>),
}

impl SExp {
    // ── Constructors ────────────────────────────────────────────────────

    /// Create an atom.
    pub fn atom(s: impl Into<String>) -> Self {
        SExp::Atom(s.into())
    }

    /// Create a list.
    pub fn list(items: Vec<SExp>) -> Self {
        SExp::List(items)
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// If this is an atom return its string.
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            SExp::Atom(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this is a list return its children.
    pub fn as_list(&self) -> Option<&[SExp]> {
        match self {
            SExp::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// True when the s-expression is an atom equal to `s`.
    pub fn is_atom(&self, s: &str) -> bool {
        self.as_atom() == Some(s)
    }

    /// True when this is a list.
    pub fn is_list(&self) -> bool {
        matches!(self, SExp::List(_))
    }

    /// Return the first child if this is a non-empty list.
    pub fn head(&self) -> Option<&SExp> {
        self.as_list().and_then(|l| l.first())
    }

    /// Return children after the first, if this is a list with ≥1 element.
    pub fn tail(&self) -> Option<&[SExp]> {
        self.as_list()
            .and_then(|l| if l.is_empty() { None } else { Some(&l[1..]) })
    }

    /// Recursively search for a list whose first atom equals `key`.
    pub fn lookup(&self, key: &str) -> Option<&SExp> {
        match self {
            SExp::List(items) => {
                if items.first().map(|h| h.is_atom(key)).unwrap_or(false) {
                    return Some(self);
                }
                for child in items {
                    if let Some(found) = child.lookup(key) {
                        return Some(found);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Number of direct children (0 for atoms).
    pub fn len(&self) -> usize {
        match self {
            SExp::Atom(_) => 0,
            SExp::List(v) => v.len(),
        }
    }

    /// Whether this is an empty list or an atom.
    pub fn is_empty(&self) -> bool {
        match self {
            SExp::Atom(_) => false,
            SExp::List(v) => v.is_empty(),
        }
    }

    /// Try to interpret an atom as an i64 (handles negative numerals and
    /// SMT-LIB `(- <numeral>)` notation).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SExp::Atom(s) => s.parse::<i64>().ok(),
            SExp::List(items) if items.len() == 2 => {
                if items[0].is_atom("-") {
                    items[1]
                        .as_atom()
                        .and_then(|s| s.parse::<i64>().ok().map(|n| -n))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to interpret as a boolean (`true` / `false`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SExp::Atom(s) if s == "true" => Some(true),
            SExp::Atom(s) if s == "false" => Some(false),
            _ => None,
        }
    }

    /// Walk all atoms in pre-order.
    pub fn atoms(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_atoms(&mut out);
        out
    }

    fn collect_atoms<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            SExp::Atom(s) => out.push(s.as_str()),
            SExp::List(items) => {
                for item in items {
                    item.collect_atoms(out);
                }
            }
        }
    }

    /// Flatten nested lists one level deep.
    pub fn flatten(&self) -> Vec<&SExp> {
        match self {
            SExp::List(items) => {
                let mut out = Vec::new();
                for item in items {
                    match item {
                        SExp::List(inner) => out.extend(inner.iter()),
                        other => out.push(other),
                    }
                }
                out
            }
            _ => vec![self],
        }
    }
}

impl fmt::Display for SExp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExp::Atom(s) => {
                if s.contains(' ') || s.contains('(') || s.contains(')') || s.is_empty() {
                    write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                } else {
                    write!(f, "{}", s)
                }
            }
            SExp::List(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ───────────────────────────── Tokenizer ─────────────────────────────────

/// Token types for the s-expression tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    LParen,
    RParen,
    Atom(String),
    QuotedString(String),
}

/// Tokenize an SMT-LIB2 response string.
fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        match chars[i] {
            // Whitespace
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            // Line comment
            ';' => {
                while i < len && chars[i] != '\n' {
                    i += 1;
                }
            }
            // Open paren
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            // Close paren
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            // Quoted string
            '"' => {
                i += 1;
                let mut s = String::new();
                loop {
                    if i >= len {
                        return Err(ParseError::UnterminatedString);
                    }
                    if chars[i] == '"' {
                        // SMT-LIB2 uses "" for escaped quote inside string
                        if i + 1 < len && chars[i + 1] == '"' {
                            s.push('"');
                            i += 2;
                        } else {
                            i += 1;
                            break;
                        }
                    } else if chars[i] == '\\' && i + 1 < len {
                        // Also handle backslash escaping used by some solvers
                        let next = chars[i + 1];
                        match next {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            '\\' => s.push('\\'),
                            '"' => s.push('"'),
                            _ => {
                                s.push('\\');
                                s.push(next);
                            }
                        }
                        i += 2;
                    } else {
                        s.push(chars[i]);
                        i += 1;
                    }
                }
                tokens.push(Token::QuotedString(s));
            }
            // Pipe-quoted symbol |...|
            '|' => {
                i += 1;
                let mut s = String::new();
                while i < len && chars[i] != '|' {
                    s.push(chars[i]);
                    i += 1;
                }
                if i >= len {
                    return Err(ParseError::UnterminatedPipeSymbol);
                }
                i += 1; // skip closing |
                tokens.push(Token::Atom(s));
            }
            // Keyword `:foo`
            ':' => {
                let start = i;
                i += 1;
                while i < len && !is_delimiter(chars[i]) {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Atom(s));
            }
            // Regular symbol / numeral
            _ => {
                let start = i;
                while i < len && !is_delimiter(chars[i]) {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Atom(s));
            }
        }
    }

    Ok(tokens)
}

fn is_delimiter(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '(' | ')' | ';' | '"' | '|')
}

// ───────────────────────────── Parser ────────────────────────────────────

/// Errors that can occur during s-expression parsing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected closing parenthesis")]
    UnexpectedRParen,
    #[error("unterminated string literal")]
    UnterminatedString,
    #[error("unterminated pipe-quoted symbol")]
    UnterminatedPipeSymbol,
    #[error("trailing tokens after complete s-expression")]
    TrailingTokens,
    #[error("empty input")]
    EmptyInput,
}

/// Parse a single s-expression from a string.
pub fn parse_sexp(input: &str) -> Result<SExp, ParseError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ParseError::EmptyInput);
    }
    let mut pos = 0;
    let result = parse_one(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(ParseError::TrailingTokens);
    }
    Ok(result)
}

/// Parse all top-level s-expressions from a string.
pub fn parse_sexp_all(input: &str) -> Result<Vec<SExp>, ParseError> {
    let tokens = tokenize(input)?;
    let mut pos = 0;
    let mut results = Vec::new();
    while pos < tokens.len() {
        results.push(parse_one(&tokens, &mut pos)?);
    }
    Ok(results)
}

/// Parse one s-expression starting at `pos`, advance `pos`.
fn parse_one(tokens: &[Token], pos: &mut usize) -> Result<SExp, ParseError> {
    if *pos >= tokens.len() {
        return Err(ParseError::UnexpectedEof);
    }
    match &tokens[*pos] {
        Token::LParen => {
            *pos += 1;
            let mut items = Vec::new();
            loop {
                if *pos >= tokens.len() {
                    return Err(ParseError::UnexpectedEof);
                }
                if tokens[*pos] == Token::RParen {
                    *pos += 1;
                    return Ok(SExp::List(items));
                }
                items.push(parse_one(tokens, pos)?);
            }
        }
        Token::RParen => Err(ParseError::UnexpectedRParen),
        Token::Atom(s) => {
            let result = SExp::Atom(s.clone());
            *pos += 1;
            Ok(result)
        }
        Token::QuotedString(s) => {
            let result = SExp::Atom(s.clone());
            *pos += 1;
            Ok(result)
        }
    }
}

// ───────────────────────── High-level parsers ────────────────────────────

/// Parse a `(model …)` or top-level `((define-fun …) …)` response into a
/// list of `(define-fun name () sort value)` entries.
pub fn parse_model_response(input: &str) -> Result<Vec<ModelEntry>, ParseError> {
    let sexps = parse_sexp_all(input)?;
    let mut entries = Vec::new();

    for sexp in &sexps {
        extract_model_entries(sexp, &mut entries);
    }

    Ok(entries)
}

/// A single `(define-fun name (params) sort body)` extracted from a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub name: String,
    pub params: Vec<SExp>,
    pub sort: SExp,
    pub body: SExp,
}

fn extract_model_entries(sexp: &SExp, out: &mut Vec<ModelEntry>) {
    if let SExp::List(items) = sexp {
        // Check if this is a define-fun
        if items.len() >= 5 {
            if let Some(head) = items[0].as_atom() {
                if head == "define-fun" {
                    if let Some(name) = items[1].as_atom() {
                        let params = match &items[2] {
                            SExp::List(ps) => ps.clone(),
                            _ => Vec::new(),
                        };
                        let sort = items[3].clone();
                        let body = items[4].clone();
                        out.push(ModelEntry {
                            name: name.to_string(),
                            params,
                            sort,
                            body,
                        });
                        return;
                    }
                }
            }
        }
        // Check if this is `(model ...)` wrapper
        if items.first().map(|h| h.is_atom("model")).unwrap_or(false) {
            for child in &items[1..] {
                extract_model_entries(child, out);
            }
            return;
        }
        // Otherwise recurse into children looking for define-fun
        for child in items {
            extract_model_entries(child, out);
        }
    }
}

/// Parse a `get-value` response: `((expr value) ...)`.
pub fn parse_get_value_response(input: &str) -> Result<Vec<(SExp, SExp)>, ParseError> {
    let sexp = parse_sexp(input)?;
    let items = sexp.as_list().ok_or(ParseError::UnexpectedEof)?;
    let mut pairs = Vec::new();
    for item in items {
        if let SExp::List(pair) = item {
            if pair.len() == 2 {
                pairs.push((pair[0].clone(), pair[1].clone()));
            }
        }
    }
    Ok(pairs)
}

/// Parse an unsat-core response: `(name1 name2 ...)`.
pub fn parse_unsat_core_response(input: &str) -> Result<Vec<String>, ParseError> {
    let sexp = parse_sexp(input)?;
    let items = sexp.as_list().ok_or(ParseError::UnexpectedEof)?;
    let mut names = Vec::new();
    for item in items {
        if let Some(name) = item.as_atom() {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

/// Parse an error response from a solver. Returns the error message string
/// if the input looks like `(error "message")`.
pub fn parse_error_response(input: &str) -> Option<String> {
    let sexp = parse_sexp(input).ok()?;
    if let SExp::List(items) = &sexp {
        if items.len() == 2 && items[0].is_atom("error") {
            return items[1].as_atom().map(|s| s.to_string());
        }
    }
    None
}

/// Parse a check-sat response string.
pub fn parse_check_sat_response(input: &str) -> CheckSatResponse {
    let trimmed = input.trim();
    match trimmed {
        "sat" => CheckSatResponse::Sat,
        "unsat" => CheckSatResponse::Unsat,
        "unknown" => CheckSatResponse::Unknown,
        "timeout" => CheckSatResponse::Timeout,
        _ => {
            if let Some(msg) = parse_error_response(trimmed) {
                CheckSatResponse::Error(msg)
            } else {
                CheckSatResponse::Error(format!("unexpected response: {}", trimmed))
            }
        }
    }
}

/// Parsed check-sat response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckSatResponse {
    Sat,
    Unsat,
    Unknown,
    Timeout,
    Error(String),
}

/// Extract integer value from an s-expression, handling SMT-LIB2 negative
/// numeral notation `(- 42)`.
pub fn sexp_to_i64(sexp: &SExp) -> Option<i64> {
    sexp.as_i64()
}

/// Extract boolean value from an s-expression.
pub fn sexp_to_bool(sexp: &SExp) -> Option<bool> {
    sexp.as_bool()
}

// ──────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic parsing ───────────────────────────────────────────────────

    #[test]
    fn test_parse_atom() {
        let sexp = parse_sexp("hello").unwrap();
        assert_eq!(sexp, SExp::atom("hello"));
    }

    #[test]
    fn test_parse_numeral() {
        let sexp = parse_sexp("42").unwrap();
        assert_eq!(sexp.as_i64(), Some(42));
    }

    #[test]
    fn test_parse_empty_list() {
        let sexp = parse_sexp("()").unwrap();
        assert_eq!(sexp, SExp::list(vec![]));
    }

    #[test]
    fn test_parse_simple_list() {
        let sexp = parse_sexp("(a b c)").unwrap();
        assert_eq!(
            sexp,
            SExp::list(vec![SExp::atom("a"), SExp::atom("b"), SExp::atom("c"),])
        );
    }

    #[test]
    fn test_parse_nested_list() {
        let sexp = parse_sexp("(a (b c) d)").unwrap();
        assert_eq!(
            sexp,
            SExp::list(vec![
                SExp::atom("a"),
                SExp::list(vec![SExp::atom("b"), SExp::atom("c")]),
                SExp::atom("d"),
            ])
        );
    }

    #[test]
    fn test_parse_quoted_string() {
        let sexp = parse_sexp(r#""hello world""#).unwrap();
        assert_eq!(sexp, SExp::atom("hello world"));
    }

    #[test]
    fn test_parse_quoted_string_with_escape() {
        let sexp = parse_sexp(r#""hello ""world""""#).unwrap();
        assert_eq!(sexp, SExp::atom(r#"hello "world""#));
    }

    #[test]
    fn test_parse_pipe_symbol() {
        let sexp = parse_sexp("|hello world|").unwrap();
        assert_eq!(sexp, SExp::atom("hello world"));
    }

    #[test]
    fn test_parse_keyword() {
        let sexp = parse_sexp(":named").unwrap();
        assert_eq!(sexp, SExp::atom(":named"));
    }

    #[test]
    fn test_parse_with_comments() {
        let input = "; this is a comment\n(a b)";
        let sexp = parse_sexp(input).unwrap();
        assert_eq!(sexp, SExp::list(vec![SExp::atom("a"), SExp::atom("b")]));
    }

    #[test]
    fn test_parse_negative_numeral() {
        let sexp = parse_sexp("(- 42)").unwrap();
        assert_eq!(sexp.as_i64(), Some(-42));
    }

    #[test]
    fn test_parse_boolean_true() {
        let sexp = parse_sexp("true").unwrap();
        assert_eq!(sexp.as_bool(), Some(true));
    }

    #[test]
    fn test_parse_boolean_false() {
        let sexp = parse_sexp("false").unwrap();
        assert_eq!(sexp.as_bool(), Some(false));
    }

    // ── Error cases ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_input() {
        assert_eq!(parse_sexp("").unwrap_err(), ParseError::EmptyInput);
    }

    #[test]
    fn test_parse_unexpected_rparen() {
        assert_eq!(parse_sexp(")").unwrap_err(), ParseError::UnexpectedRParen);
    }

    #[test]
    fn test_parse_unterminated_list() {
        assert_eq!(parse_sexp("(a b").unwrap_err(), ParseError::UnexpectedEof);
    }

    #[test]
    fn test_parse_unterminated_string() {
        assert_eq!(
            parse_sexp("\"hello").unwrap_err(),
            ParseError::UnterminatedString
        );
    }

    #[test]
    fn test_parse_trailing_tokens() {
        assert_eq!(parse_sexp("a b").unwrap_err(), ParseError::TrailingTokens);
    }

    // ── parse_sexp_all ──────────────────────────────────────────────────

    #[test]
    fn test_parse_multiple_sexps() {
        let sexps = parse_sexp_all("(a b) (c d) e").unwrap();
        assert_eq!(sexps.len(), 3);
        assert!(sexps[0].is_list());
        assert!(sexps[1].is_list());
        assert!(sexps[2].is_atom("e"));
    }

    // ── Model parsing ───────────────────────────────────────────────────

    #[test]
    fn test_parse_model_define_fun() {
        let input = r#"(model
  (define-fun x () Int 42)
  (define-fun y () Int (- 7))
  (define-fun b () Bool true)
)"#;
        let entries = parse_model_response(input).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "x");
        assert_eq!(entries[0].body.as_i64(), Some(42));
        assert_eq!(entries[1].name, "y");
        assert_eq!(entries[1].body.as_i64(), Some(-7));
        assert_eq!(entries[2].name, "b");
        assert_eq!(entries[2].body.as_bool(), Some(true));
    }

    #[test]
    fn test_parse_model_without_model_wrapper() {
        let input = r#"(
  (define-fun x () Int 10)
  (define-fun y () Int 20)
)"#;
        let entries = parse_model_response(input).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_parse_model_flat() {
        let input = "(define-fun x () Int 42)";
        let entries = parse_model_response(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "x");
    }

    // ── get-value parsing ───────────────────────────────────────────────

    #[test]
    fn test_parse_get_value() {
        let input = "((x 42) (y (- 3)))";
        let pairs = parse_get_value_response(input).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, SExp::atom("x"));
        assert_eq!(pairs[0].1.as_i64(), Some(42));
        assert_eq!(pairs[1].1.as_i64(), Some(-3));
    }

    // ── Unsat core parsing ──────────────────────────────────────────────

    #[test]
    fn test_parse_unsat_core() {
        let input = "(a1 a2 a3)";
        let names = parse_unsat_core_response(input).unwrap();
        assert_eq!(names, vec!["a1", "a2", "a3"]);
    }

    #[test]
    fn test_parse_unsat_core_empty() {
        let input = "()";
        let names = parse_unsat_core_response(input).unwrap();
        assert!(names.is_empty());
    }

    // ── Error response parsing ──────────────────────────────────────────

    #[test]
    fn test_parse_error_response() {
        let input = r#"(error "line 3: unknown sort")"#;
        let msg = parse_error_response(input);
        assert_eq!(msg, Some("line 3: unknown sort".to_string()));
    }

    #[test]
    fn test_parse_error_response_not_error() {
        assert_eq!(parse_error_response("sat"), None);
    }

    // ── check-sat parsing ───────────────────────────────────────────────

    #[test]
    fn test_check_sat_sat() {
        assert_eq!(parse_check_sat_response("sat"), CheckSatResponse::Sat);
    }

    #[test]
    fn test_check_sat_unsat() {
        assert_eq!(parse_check_sat_response("unsat"), CheckSatResponse::Unsat);
    }

    #[test]
    fn test_check_sat_unknown() {
        assert_eq!(
            parse_check_sat_response("unknown"),
            CheckSatResponse::Unknown
        );
    }

    #[test]
    fn test_check_sat_error() {
        let resp = parse_check_sat_response(r#"(error "oops")"#);
        assert_eq!(resp, CheckSatResponse::Error("oops".to_string()));
    }

    #[test]
    fn test_check_sat_unexpected() {
        let resp = parse_check_sat_response("gibberish");
        assert!(matches!(resp, CheckSatResponse::Error(_)));
    }

    // ── SExp utility methods ────────────────────────────────────────────

    #[test]
    fn test_sexp_lookup() {
        let sexp = parse_sexp("(a (foo 1 2) (bar 3))").unwrap();
        let found = sexp.lookup("foo").unwrap();
        assert_eq!(
            found,
            &SExp::list(vec![SExp::atom("foo"), SExp::atom("1"), SExp::atom("2"),])
        );
    }

    #[test]
    fn test_sexp_head_tail() {
        let sexp = parse_sexp("(a b c)").unwrap();
        assert_eq!(sexp.head(), Some(&SExp::atom("a")));
        let tail = sexp.tail().unwrap();
        assert_eq!(tail.len(), 2);
    }

    #[test]
    fn test_sexp_atoms() {
        let sexp = parse_sexp("(a (b c) d)").unwrap();
        assert_eq!(sexp.atoms(), vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_sexp_display_atom() {
        assert_eq!(SExp::atom("hello").to_string(), "hello");
    }

    #[test]
    fn test_sexp_display_list() {
        let sexp = SExp::list(vec![SExp::atom("+"), SExp::atom("x"), SExp::atom("1")]);
        assert_eq!(sexp.to_string(), "(+ x 1)");
    }

    #[test]
    fn test_sexp_display_quoted() {
        let sexp = SExp::atom("hello world");
        assert_eq!(sexp.to_string(), "\"hello world\"");
    }

    // ── Complex model formats ───────────────────────────────────────────

    #[test]
    fn test_parse_z3_model_with_array() {
        let input = r#"(model
  (define-fun a () (Array Int Int) ((as const (Array Int Int)) 0))
)"#;
        let entries = parse_model_response(input).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "a");
    }

    #[test]
    fn test_parse_model_multiline() {
        let input = r#"(model
  (define-fun x () Int
    42)
  (define-fun y () Int
    (- 3))
)"#;
        let entries = parse_model_response(input).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].body.as_i64(), Some(42));
        assert_eq!(entries[1].body.as_i64(), Some(-3));
    }

    #[test]
    fn test_parse_deeply_nested() {
        let input = "(a (b (c (d e))))";
        let sexp = parse_sexp(input).unwrap();
        let d_list = sexp.lookup("d").expect("should find (d e)");
        assert_eq!(d_list, &SExp::list(vec![SExp::atom("d"), SExp::atom("e")]));
    }

    #[test]
    fn test_roundtrip_display_parse() {
        let original = SExp::list(vec![
            SExp::atom("define-fun"),
            SExp::atom("x"),
            SExp::list(vec![]),
            SExp::atom("Int"),
            SExp::list(vec![SExp::atom("-"), SExp::atom("42")]),
        ]);
        let printed = original.to_string();
        let parsed = parse_sexp(&printed).unwrap();
        assert_eq!(original, parsed);
    }
}
