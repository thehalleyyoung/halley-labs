//! EDN (Extensible Data Notation) parser for Jepsen history import.
//!
//! Jepsen uses EDN (a subset of Clojure syntax) to represent operation histories.
//! This module provides a minimal EDN parser sufficient to read Jepsen history files.
//!
//! EDN spec: <https://github.com/edn-format/edn>

use crate::{FormatError, FormatResult};
use std::collections::HashMap;
use std::fmt;

/// An EDN value
#[derive(Debug, Clone, PartialEq)]
pub enum EdnValue {
    Nil,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Keyword(String),
    Symbol(String),
    Vector(Vec<EdnValue>),
    List(Vec<EdnValue>),
    Map(Vec<(EdnValue, EdnValue)>),
    Set(Vec<EdnValue>),
    Tagged(String, Box<EdnValue>),
}

impl EdnValue {
    /// Get as keyword string (without leading colon)
    pub fn as_keyword(&self) -> Option<&str> {
        match self {
            EdnValue::Keyword(k) => Some(k.as_str()),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            EdnValue::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            EdnValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get as vector
    pub fn as_vector(&self) -> Option<&[EdnValue]> {
        match self {
            EdnValue::Vector(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get as map
    pub fn as_map(&self) -> Option<&[(EdnValue, EdnValue)]> {
        match self {
            EdnValue::Map(m) => Some(m.as_slice()),
            _ => None,
        }
    }

    /// Lookup a key in a map by keyword name
    pub fn get(&self, key: &str) -> Option<&EdnValue> {
        match self {
            EdnValue::Map(pairs) => pairs.iter().find_map(|(k, v)| {
                if k.as_keyword() == Some(key) {
                    Some(v)
                } else {
                    None
                }
            }),
            _ => None,
        }
    }
}

impl fmt::Display for EdnValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdnValue::Nil => write!(f, "nil"),
            EdnValue::Bool(b) => write!(f, "{}", b),
            EdnValue::Integer(n) => write!(f, "{}", n),
            EdnValue::Float(n) => write!(f, "{}", n),
            EdnValue::String(s) => write!(f, "\"{}\"", s),
            EdnValue::Keyword(k) => write!(f, ":{}", k),
            EdnValue::Symbol(s) => write!(f, "{}", s),
            EdnValue::Vector(v) => {
                write!(f, "[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            EdnValue::List(v) => {
                write!(f, "(")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            EdnValue::Map(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} {}", k, v)?;
                }
                write!(f, "}}")
            }
            EdnValue::Set(v) => {
                write!(f, "#{{")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "}}")
            }
            EdnValue::Tagged(tag, val) => write!(f, "#{} {}", tag, val),
        }
    }
}

/// Minimal EDN parser
pub struct EdnParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> EdnParser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    /// Parse the entire input as a sequence of EDN values.
    pub fn parse_all(&mut self) -> FormatResult<Vec<EdnValue>> {
        let mut values = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                break;
            }
            values.push(self.parse_value()?);
        }
        Ok(values)
    }

    /// Parse a single EDN value.
    pub fn parse_value(&mut self) -> FormatResult<EdnValue> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Err(FormatError::EdnSyntaxError {
                position: self.pos,
                detail: "Unexpected end of input".into(),
            });
        }

        match self.input[self.pos] {
            b'"' => self.parse_string(),
            b':' => self.parse_keyword(),
            b'[' => self.parse_vector(),
            b'(' => self.parse_list(),
            b'{' => self.parse_map(),
            b'#' => self.parse_dispatch(),
            b'-' | b'+' | b'0'..=b'9' => self.parse_number(),
            _ => self.parse_symbol_or_bool(),
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' | b',' => self.pos += 1,
                b';' => {
                    // Line comment
                    while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn parse_string(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip opening quote
        let mut s = String::new();
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b'"' => {
                    self.pos += 1;
                    return Ok(EdnValue::String(s));
                }
                b'\\' => {
                    self.pos += 1;
                    if self.pos >= self.input.len() {
                        return Err(FormatError::EdnSyntaxError {
                            position: self.pos,
                            detail: "Unexpected end in string escape".into(),
                        });
                    }
                    match self.input[self.pos] {
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        c => s.push(c as char),
                    }
                    self.pos += 1;
                }
                c => {
                    s.push(c as char);
                    self.pos += 1;
                }
            }
        }
        Err(FormatError::EdnSyntaxError {
            position: self.pos,
            detail: "Unterminated string".into(),
        })
    }

    fn parse_keyword(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip colon
        let start = self.pos;
        while self.pos < self.input.len() && is_symbol_char(self.input[self.pos]) {
            self.pos += 1;
        }
        let name = String::from_utf8_lossy(&self.input[start..self.pos]).to_string();
        Ok(EdnValue::Keyword(name))
    }

    fn parse_number(&mut self) -> FormatResult<EdnValue> {
        let start = self.pos;
        if self.input[self.pos] == b'-' || self.input[self.pos] == b'+' {
            self.pos += 1;
        }
        let mut has_dot = false;
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b'0'..=b'9' => self.pos += 1,
                b'.' => {
                    has_dot = true;
                    self.pos += 1;
                }
                b'E' | b'e' => {
                    has_dot = true;
                    self.pos += 1;
                    if self.pos < self.input.len()
                        && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
                    {
                        self.pos += 1;
                    }
                }
                b'N' => {
                    // BigInteger suffix
                    self.pos += 1;
                    break;
                }
                b'M' => {
                    // BigDecimal suffix
                    has_dot = true;
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }

        let text = String::from_utf8_lossy(&self.input[start..self.pos]);
        let text = text.trim_end_matches(|c| c == 'N' || c == 'M');

        if has_dot {
            text.parse::<f64>()
                .map(EdnValue::Float)
                .map_err(|_| FormatError::EdnSyntaxError {
                    position: start,
                    detail: format!("Invalid float: {}", text),
                })
        } else {
            text.parse::<i64>()
                .map(EdnValue::Integer)
                .map_err(|_| FormatError::EdnSyntaxError {
                    position: start,
                    detail: format!("Invalid integer: {}", text),
                })
        }
    }

    fn parse_vector(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip [
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                return Err(FormatError::EdnSyntaxError {
                    position: self.pos,
                    detail: "Unterminated vector".into(),
                });
            }
            if self.input[self.pos] == b']' {
                self.pos += 1;
                return Ok(EdnValue::Vector(items));
            }
            items.push(self.parse_value()?);
        }
    }

    fn parse_list(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip (
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                return Err(FormatError::EdnSyntaxError {
                    position: self.pos,
                    detail: "Unterminated list".into(),
                });
            }
            if self.input[self.pos] == b')' {
                self.pos += 1;
                return Ok(EdnValue::List(items));
            }
            items.push(self.parse_value()?);
        }
    }

    fn parse_map(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip {
        let mut pairs = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                return Err(FormatError::EdnSyntaxError {
                    position: self.pos,
                    detail: "Unterminated map".into(),
                });
            }
            if self.input[self.pos] == b'}' {
                self.pos += 1;
                return Ok(EdnValue::Map(pairs));
            }
            let key = self.parse_value()?;
            let val = self.parse_value()?;
            pairs.push((key, val));
        }
    }

    fn parse_dispatch(&mut self) -> FormatResult<EdnValue> {
        self.pos += 1; // skip #
        if self.pos >= self.input.len() {
            return Err(FormatError::EdnSyntaxError {
                position: self.pos,
                detail: "Unexpected end after #".into(),
            });
        }

        match self.input[self.pos] {
            b'{' => {
                // Set literal #{...}
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_whitespace_and_comments();
                    if self.pos >= self.input.len() {
                        return Err(FormatError::EdnSyntaxError {
                            position: self.pos,
                            detail: "Unterminated set".into(),
                        });
                    }
                    if self.input[self.pos] == b'}' {
                        self.pos += 1;
                        return Ok(EdnValue::Set(items));
                    }
                    items.push(self.parse_value()?);
                }
            }
            b'_' => {
                // Discard: #_ value
                self.pos += 1;
                let _ = self.parse_value()?;
                self.parse_value()
            }
            _ => {
                // Tagged literal: #tag value
                let start = self.pos;
                while self.pos < self.input.len() && is_symbol_char(self.input[self.pos]) {
                    self.pos += 1;
                }
                let tag = String::from_utf8_lossy(&self.input[start..self.pos]).to_string();
                self.skip_whitespace_and_comments();
                let value = self.parse_value()?;
                Ok(EdnValue::Tagged(tag, Box::new(value)))
            }
        }
    }

    fn parse_symbol_or_bool(&mut self) -> FormatResult<EdnValue> {
        let start = self.pos;
        while self.pos < self.input.len() && is_symbol_char(self.input[self.pos]) {
            self.pos += 1;
        }
        let text = String::from_utf8_lossy(&self.input[start..self.pos]).to_string();
        match text.as_str() {
            "nil" => Ok(EdnValue::Nil),
            "true" => Ok(EdnValue::Bool(true)),
            "false" => Ok(EdnValue::Bool(false)),
            _ => Ok(EdnValue::Symbol(text)),
        }
    }
}

fn is_symbol_char(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'
        | b'.' | b'*' | b'+' | b'!' | b'-' | b'_' | b'?'
        | b'$' | b'%' | b'&' | b'=' | b'<' | b'>' | b'/' | b'\'')
}

/// Serialize an EdnValue to string.
pub fn to_edn_string(value: &EdnValue) -> String {
    format!("{}", value)
}

/// Parse an EDN string into values.
pub fn parse_edn(input: &str) -> FormatResult<Vec<EdnValue>> {
    EdnParser::new(input).parse_all()
}

/// Parse a single EDN value from a string.
pub fn parse_edn_value(input: &str) -> FormatResult<EdnValue> {
    EdnParser::new(input).parse_value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nil() {
        assert_eq!(parse_edn_value("nil").unwrap(), EdnValue::Nil);
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse_edn_value("true").unwrap(), EdnValue::Bool(true));
        assert_eq!(parse_edn_value("false").unwrap(), EdnValue::Bool(false));
    }

    #[test]
    fn test_parse_integer() {
        assert_eq!(parse_edn_value("42").unwrap(), EdnValue::Integer(42));
        assert_eq!(parse_edn_value("-7").unwrap(), EdnValue::Integer(-7));
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(parse_edn_value("3.14").unwrap(), EdnValue::Float(3.14));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            parse_edn_value("\"hello\"").unwrap(),
            EdnValue::String("hello".into())
        );
        assert_eq!(
            parse_edn_value("\"line\\nnewline\"").unwrap(),
            EdnValue::String("line\nnewline".into())
        );
    }

    #[test]
    fn test_parse_keyword() {
        assert_eq!(
            parse_edn_value(":ok").unwrap(),
            EdnValue::Keyword("ok".into())
        );
        assert_eq!(
            parse_edn_value(":txn/read").unwrap(),
            EdnValue::Keyword("txn/read".into())
        );
    }

    #[test]
    fn test_parse_vector() {
        let v = parse_edn_value("[1 2 3]").unwrap();
        assert_eq!(
            v,
            EdnValue::Vector(vec![
                EdnValue::Integer(1),
                EdnValue::Integer(2),
                EdnValue::Integer(3),
            ])
        );
    }

    #[test]
    fn test_parse_map() {
        let v = parse_edn_value("{:type :ok :value 42}").unwrap();
        match v {
            EdnValue::Map(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert_eq!(pairs[0].0, EdnValue::Keyword("type".into()));
                assert_eq!(pairs[0].1, EdnValue::Keyword("ok".into()));
                assert_eq!(pairs[1].0, EdnValue::Keyword("value".into()));
                assert_eq!(pairs[1].1, EdnValue::Integer(42));
            }
            _ => panic!("Expected map"),
        }
    }

    #[test]
    fn test_parse_set() {
        let v = parse_edn_value("#{1 2 3}").unwrap();
        match v {
            EdnValue::Set(items) => assert_eq!(items.len(), 3),
            _ => panic!("Expected set"),
        }
    }

    #[test]
    fn test_parse_nested() {
        let input = r#"{:type :invoke :f :txn :value [[:r :x nil] [:w :y 1]]}"#;
        let v = parse_edn_value(input).unwrap();
        assert!(v.get("type").is_some());
        assert_eq!(v.get("type").unwrap().as_keyword(), Some("invoke"));
    }

    #[test]
    fn test_comments() {
        let input = "; comment\n42 ; trailing";
        let values = parse_edn(input).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], EdnValue::Integer(42));
    }

    #[test]
    fn test_jepsen_operation() {
        let input = r#"{:type :ok, :f :txn, :value [[:r :x 1] [:w :y 2]], :process 0, :time 12345}"#;
        let v = parse_edn_value(input).unwrap();
        assert_eq!(v.get("type").unwrap().as_keyword(), Some("ok"));
        assert_eq!(v.get("f").unwrap().as_keyword(), Some("txn"));
        assert_eq!(v.get("process").unwrap().as_integer(), Some(0));
    }
}
