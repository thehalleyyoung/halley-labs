//! Litmus test parser supporting multiple formats.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};
use crate::checker::litmus::*;
use crate::checker::execution::*;

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Number(i64),
    String(String),
    LParen,
    RParen,
    LBrace,
    LBracket,
    RBracket,
    RBrace,
    Semicolon,
    Colon,
    Comma,
    Equals,
    Pipe,
    Arrow,
    At,
    Hash,
    Star,
    Plus,
    Tilde,
    Newline,
    Eof,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0, line: 1, col: 1 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        if ch == '\n' { self.line += 1; self.col = 1; } else { self.col += 1; }
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance_char();
            } else if ch == '/' {
                // Skip line comments
                let rest = &self.input[self.pos..];
                if rest.starts_with("//") {
                    while let Some(c) = self.peek_char() {
                        if c == '\n' { break; }
                        self.advance_char();
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                self.advance_char();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn read_number(&mut self) -> i64 {
        let start = self.pos;
        if self.peek_char() == Some('-') { self.advance_char(); }
        let rest = &self.input[self.pos..];
        if rest.starts_with("0x") || rest.starts_with("0X") {
            self.advance_char(); self.advance_char();
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_hexdigit() { self.advance_char(); } else { break; }
            }
            let hex_str = &self.input[start..self.pos];
            i64::from_str_radix(hex_str.trim_start_matches('-').trim_start_matches("0x").trim_start_matches("0X"), 16)
                .unwrap_or(0)
        } else {
            while let Some(ch) = self.peek_char() {
                if ch.is_ascii_digit() { self.advance_char(); } else { break; }
            }
            self.input[start..self.pos].parse().unwrap_or(0)
        }
    }

    fn read_string(&mut self) -> String {
        self.advance_char(); // skip opening quote
        let mut s = String::new();
        while let Some(ch) = self.advance_char() {
            if ch == '"' { break; }
            if ch == '\\' {
                if let Some(esc) = self.advance_char() {
                    match esc {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        _ => { s.push('\\'); s.push(esc); }
                    }
                }
            } else {
                s.push(ch);
            }
        }
        s
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek_char() {
                None => { tokens.push(Token::Eof); break; }
                Some('\n') => { self.advance_char(); tokens.push(Token::Newline); }
                Some('(') => { self.advance_char(); tokens.push(Token::LParen); }
                Some(')') => { self.advance_char(); tokens.push(Token::RParen); }
                Some('{') => { self.advance_char(); tokens.push(Token::LBrace); }
                Some('}') => { self.advance_char(); tokens.push(Token::RBrace); }
                Some('[') => { self.advance_char(); tokens.push(Token::LBracket); }
                Some(']') => { self.advance_char(); tokens.push(Token::RBracket); }
                Some(';') => { self.advance_char(); tokens.push(Token::Semicolon); }
                Some(':') => { self.advance_char(); tokens.push(Token::Colon); }
                Some(',') => { self.advance_char(); tokens.push(Token::Comma); }
                Some('=') => { self.advance_char(); tokens.push(Token::Equals); }
                Some('|') => { self.advance_char(); tokens.push(Token::Pipe); }
                Some('@') => { self.advance_char(); tokens.push(Token::At); }
                Some('#') => { self.advance_char(); tokens.push(Token::Hash); }
                Some('*') => { self.advance_char(); tokens.push(Token::Star); }
                Some('+') => { self.advance_char(); tokens.push(Token::Plus); }
                Some('~') => { self.advance_char(); tokens.push(Token::Tilde); }
                Some('"') => { let s = self.read_string(); tokens.push(Token::String(s)); }
                Some(ch) if ch.is_ascii_digit() => {
                    let n = self.read_number();
                    tokens.push(Token::Number(n));
                }
                Some('-') => {
                    let rest = &self.input[self.pos..];
                    if rest.len() > 1 && rest.as_bytes()[1].is_ascii_digit() {
                        let n = self.read_number();
                        tokens.push(Token::Number(n));
                    } else if rest.starts_with("->") {
                        self.advance_char(); self.advance_char();
                        tokens.push(Token::Arrow);
                    } else {
                        let ident = self.read_ident();
                        tokens.push(Token::Ident(ident));
                    }
                }
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    let ident = self.read_ident();
                    tokens.push(Token::Ident(ident));
                }
                Some(_) => { self.advance_char(); }
            }
        }
        tokens
    }
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// LitmusParser
// ---------------------------------------------------------------------------

/// Parser for litmus test files, supporting multiple formats.
pub struct LitmusParser;

impl LitmusParser {
    pub fn new() -> Self { LitmusParser }

    /// Auto-detect format and parse.
    pub fn parse(&self, input: &str) -> Result<LitmusTest, ParseError> {
        let trimmed = input.trim();
        if trimmed.starts_with('{') || trimmed.contains("exists") {
            self.parse_herd(input)
        } else if trimmed.starts_with("LISA") || trimmed.contains("LISA") {
            self.parse_lisa(input)
        } else if trimmed.contains("ld.") || trimmed.contains("st.") || trimmed.contains(".global") {
            self.parse_ptx(input)
        } else {
            self.parse_simple(input)
        }
    }

    /// Parse Herd-style litmus test format.
    pub fn parse_herd(&self, input: &str) -> Result<LitmusTest, ParseError> {
        let mut test = LitmusTest::new("unnamed");
        let lines: Vec<&str> = input.lines().collect();
        let mut i = 0;

        // Parse header: architecture and test name
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() || line.starts_with("//") || line.starts_with("(*") {
                i += 1;
                continue;
            }
            // Check for architecture line: "PTX SB" or "C SB"
            if !line.starts_with('{') && !line.starts_with("P") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    test.name = parts[1..].join(" ");
                }
                i += 1;
                break;
            }
            break;
        }

        // Parse initial state { x=0; y=0; }
        while i < lines.len() {
            let line = lines[i].trim();
            if line.starts_with('{') {
                let block = self.extract_block(&lines, &mut i, '{', '}');
                for stmt in block.split(';') {
                    let stmt = stmt.trim();
                    if stmt.is_empty() { continue; }
                    let parts: Vec<&str> = stmt.split('=').collect();
                    if parts.len() == 2 {
                        let var = parts[0].trim();
                        let val: Value = parts[1].trim().parse().unwrap_or(0);
                        test.initial_state.insert(self.addr_from_name(var), val);
                    }
                }
                i += 1;
                break;
            }
            i += 1;
        }

        // Parse threads
        let mut current_thread: Option<Thread> = None;
        while i < lines.len() {
            let line = lines[i].trim();
            i += 1;

            if line.is_empty() || line.starts_with("//") { continue; }
            if line.starts_with("exists") || line.starts_with("forall") || line.starts_with("~exists") {
                // Parse outcome
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let outcome_str = self.extract_outcome_block(&lines, i - 1);
                if let Some(outcome) = self.parse_outcome(&outcome_str) {
                    if line.starts_with("~exists") || line.starts_with("forall") {
                        test.expected_outcomes.push((outcome, LitmusOutcome::Forbidden));
                    } else {
                        test.expected_outcomes.push((outcome, LitmusOutcome::Allowed));
                    }
                }
                break;
            }

            // Thread header: "P0:" or "P0 |" etc.
            if line.starts_with('P') || line.starts_with('T') {
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let tid = self.extract_thread_id(line);
                current_thread = Some(Thread::new(tid));

                // Instructions on same line after ':'
                if let Some(colon_pos) = line.find(':') {
                    let inst_str = &line[colon_pos + 1..];
                    if let Some(ref mut t) = current_thread {
                        self.parse_instructions(inst_str, t);
                    }
                }
                continue;
            }

            // Parse instruction lines
            if let Some(ref mut t) = current_thread {
                // Skip thread separator '|'
                let clean = line.trim_start_matches('|').trim();
                if !clean.is_empty() {
                    self.parse_instructions(clean, t);
                }
            }
        }

        if let Some(t) = current_thread {
            test.add_thread(t);
        }

        if test.threads.is_empty() {
            return Err(ParseError { message: "No threads found".to_string(), line: 0, col: 0 });
        }

        Ok(test)
    }

    /// Parse LISA format.
    pub fn parse_lisa(&self, input: &str) -> Result<LitmusTest, ParseError> {
        let mut test = LitmusTest::new("LISA test");
        let lines: Vec<&str> = input.lines().collect();
        let mut current_thread: Option<Thread> = None;

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("LISA") { continue; }

            if trimmed.starts_with("thread") || trimmed.starts_with("Thread") {
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let tid = self.extract_thread_id(trimmed);
                current_thread = Some(Thread::new(tid));
                continue;
            }

            if let Some(ref mut t) = current_thread {
                // LISA instructions: "w[x] 1" or "r[x] r0"
                if trimmed.starts_with("w[") {
                    if let Some((addr, val)) = self.parse_lisa_write(trimmed) {
                        t.add(Instruction::Store { addr: self.addr_from_name(&addr), value: val, ordering: Ordering::Relaxed });
                    }
                } else if trimmed.starts_with("r[") {
                    if let Some((addr, reg)) = self.parse_lisa_read(trimmed) {
                        t.add(Instruction::Load { reg: self.reg_from_name(&reg), addr: self.addr_from_name(&addr), ordering: Ordering::Relaxed });
                    }
                } else if trimmed.starts_with("f[") || trimmed == "fence" {
                    t.add(Instruction::Fence { ordering: Ordering::SeqCst, scope: crate::checker::litmus::Scope::None });
                }
            }
        }

        if let Some(t) = current_thread {
            test.add_thread(t);
        }

        Ok(test)
    }

    /// Parse PTX assembly format.
    pub fn parse_ptx(&self, input: &str) -> Result<LitmusTest, ParseError> {
        let mut test = LitmusTest::new("PTX test");
        let lines: Vec<&str> = input.lines().collect();
        let mut current_thread: Option<Thread> = None;

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") { continue; }

            if trimmed.starts_with("thread") || trimmed.starts_with("Thread") || trimmed.starts_with("P") {
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let tid = self.extract_thread_id(trimmed);
                current_thread = Some(Thread::new(tid));
                continue;
            }

            if let Some(ref mut t) = current_thread {
                // PTX: "st.global.relaxed.sys [x], 1"
                if trimmed.starts_with("st.") {
                    let ordering = self.parse_ptx_ordering(trimmed);
                    if let Some((addr, val)) = self.parse_ptx_store(trimmed) {
                        t.add(Instruction::Store { addr: self.addr_from_name(&addr), value: val, ordering });
                    }
                }
                // PTX: "ld.global.acquire.sys r0, [x]"
                else if trimmed.starts_with("ld.") {
                    let ordering = self.parse_ptx_ordering(trimmed);
                    if let Some((reg, addr)) = self.parse_ptx_load(trimmed) {
                        t.add(Instruction::Load { reg: self.reg_from_name(&reg), addr: self.addr_from_name(&addr), ordering });
                    }
                }
                // PTX: "membar.sys" or "fence.sc.sys"
                else if trimmed.starts_with("membar") || trimmed.starts_with("fence") {
                    let scope = if trimmed.contains(".cta") { crate::checker::litmus::Scope::CTA }
                        else if trimmed.contains(".gpu") { crate::checker::litmus::Scope::GPU }
                        else { crate::checker::litmus::Scope::System };
                    t.add(Instruction::Fence { ordering: Ordering::SeqCst, scope });
                }
            }
        }

        if let Some(t) = current_thread {
            test.add_thread(t);
        }

        Ok(test)
    }

    /// Parse simplified custom format.
    pub fn parse_simple(&self, input: &str) -> Result<LitmusTest, ParseError> {
        let mut test = LitmusTest::new("test");
        let lines: Vec<&str> = input.lines().collect();
        let mut current_thread: Option<Thread> = None;

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") { continue; }

            // Test name
            if trimmed.starts_with("test") || trimmed.starts_with("name") {
                if let Some(name) = trimmed.split_whitespace().nth(1) {
                    test.name = name.to_string();
                }
                continue;
            }

            // Thread header
            if trimmed.starts_with("thread") || trimmed.starts_with("T") {
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let tid = self.extract_thread_id(trimmed);
                current_thread = Some(Thread::new(tid));
                continue;
            }

            // Outcome
            if trimmed.starts_with("forbidden") || trimmed.starts_with("allowed") || trimmed.starts_with("required") {
                if let Some(t) = current_thread.take() {
                    test.add_thread(t);
                }
                let rest = trimmed.splitn(2, ':').nth(1).unwrap_or("").trim();
                if let Some(outcome) = self.parse_outcome(rest) {
                    if trimmed.starts_with("forbidden") {
                        test.expected_outcomes.push((outcome, LitmusOutcome::Forbidden));
                    } else if trimmed.starts_with("required") {
                        test.expected_outcomes.push((outcome, LitmusOutcome::Required));
                    } else {
                        test.expected_outcomes.push((outcome, LitmusOutcome::Allowed));
                    }
                }
                continue;
            }

            // Instructions
            if let Some(ref mut t) = current_thread {
                self.parse_instructions(trimmed, t);
            }
        }

        if let Some(t) = current_thread {
            test.add_thread(t);
        }

        Ok(test)
    }

    // --- Helper methods ---

    fn extract_block(&self, lines: &[&str], i: &mut usize, open: char, close: char) -> String {
        let mut block = String::new();
        let mut depth = 0;
        while *i < lines.len() {
            let line = lines[*i];
            for ch in line.chars() {
                if ch == open { depth += 1; }
                else if ch == close {
                    depth -= 1;
                    if depth == 0 { return block; }
                }
                else if depth > 0 { block.push(ch); }
            }
            if depth > 0 { block.push(' '); }
            *i += 1;
        }
        block
    }

    fn extract_outcome_block(&self, lines: &[&str], start: usize) -> String {
        let mut result = String::new();
        for i in start..lines.len() {
            result.push_str(lines[i]);
            result.push(' ');
            if lines[i].contains(')') { break; }
        }
        result
    }

    fn extract_thread_id(&self, line: &str) -> ThreadId {
        for word in line.split(|c: char| !c.is_ascii_digit()) {
            if let Ok(id) = word.parse::<ThreadId>() {
                return id;
            }
        }
        0
    }

    fn parse_instructions(&self, text: &str, thread: &mut Thread) {
        for part in text.split(';') {
            let part = part.trim();
            if part.is_empty() { continue; }

            let lower = part.to_lowercase();
            if lower.starts_with("w(") || lower.starts_with("w ") || lower.starts_with("write") {
                // W(x, 1) or W x 1
                if let Some((addr, val)) = self.parse_write_inst(part) {
                    thread.add(Instruction::Store { addr: self.addr_from_name(&addr), value: val, ordering: Ordering::Relaxed });
                }
            } else if lower.starts_with("r(") || lower.starts_with("r ") || lower.starts_with("read") {
                // R(x) r0 or R x r0
                if let Some((addr, reg)) = self.parse_read_inst(part) {
                    thread.add(Instruction::Load { reg: self.reg_from_name(&reg), addr: self.addr_from_name(&addr), ordering: Ordering::Relaxed });
                }
            } else if lower.starts_with("f") || lower.contains("fence") {
                thread.add(Instruction::Fence { ordering: Ordering::SeqCst, scope: crate::checker::litmus::Scope::None });
            }
        }
    }

    fn parse_write_inst(&self, text: &str) -> Option<(String, Value)> {
        let clean = text.replace(|c: char| c == '(' || c == ')' || c == ',', " ");
        let parts: Vec<&str> = clean.split_whitespace().collect();
        if parts.len() >= 3 {
            let addr = parts[1].to_string();
            let val: Value = parts[2].parse().unwrap_or(0);
            Some((addr, val))
        } else if parts.len() == 2 {
            Some((parts[1].to_string(), 1))
        } else {
            None
        }
    }

    fn parse_read_inst(&self, text: &str) -> Option<(String, String)> {
        let clean = text.replace(|c: char| c == '(' || c == ')' || c == ',', " ");
        let parts: Vec<&str> = clean.split_whitespace().collect();
        if parts.len() >= 3 {
            Some((parts[1].to_string(), parts[2].to_string()))
        } else if parts.len() == 2 {
            Some((parts[1].to_string(), format!("r_{}", parts[1])))
        } else {
            None
        }
    }

    fn parse_outcome(&self, text: &str) -> Option<Outcome> {
        let clean = text.replace(|c: char| c == '(' || c == ')' || c == '/', ",")
            .replace("\\", " ")
            .replace("&", " ");
        let clean = clean.replace("/\\", " ");
        let mut outcome = Outcome::new();
        let mut found = false;
        for part in clean.split(|c: char| c == ',' || c == ';' || c == ' ') {
            let part = part.trim();
            if part.contains('=') {
                let kv: Vec<&str> = part.split('=').collect();
                if kv.len() == 2 {
                    let key = kv[0].trim().trim_matches(':').to_string();
                    if let Ok(val) = kv[1].trim().parse::<Value>() {
                        outcome.memory.insert(self.addr_from_name(&key), val);
                        found = true;
                    }
                }
            }
        }
        if found { Some(outcome) } else { None }
    }

    fn parse_lisa_write(&self, text: &str) -> Option<(String, Value)> {
        // "w[x] 1"
        let start = text.find('[')? + 1;
        let end = text.find(']')?;
        let addr = text[start..end].trim().to_string();
        let rest = text[end + 1..].trim();
        let val: Value = rest.parse().unwrap_or(1);
        Some((addr, val))
    }

    fn parse_lisa_read(&self, text: &str) -> Option<(String, String)> {
        // "r[x] r0"
        let start = text.find('[')? + 1;
        let end = text.find(']')?;
        let addr = text[start..end].trim().to_string();
        let rest = text[end + 1..].trim();
        let reg = if rest.is_empty() { format!("r_{}", addr) } else { rest.to_string() };
        Some((addr, reg))
    }

    fn parse_ptx_ordering(&self, text: &str) -> Ordering {
        if text.contains(".relaxed") { Ordering::Relaxed }
        else if text.contains(".acquire") { Ordering::Acquire }
        else if text.contains(".release") { Ordering::Release }
        else if text.contains(".acq_rel") { Ordering::AcqRel }
        else if text.contains(".sc") { Ordering::SeqCst }
        else { Ordering::Relaxed }
    }

    fn parse_ptx_store(&self, text: &str) -> Option<(String, Value)> {
        // "st.global.relaxed.sys [x], 1"
        let bracket_start = text.find('[')? + 1;
        let bracket_end = text.find(']')?;
        let addr = text[bracket_start..bracket_end].trim().to_string();
        let after = text[bracket_end + 1..].trim_start_matches(',').trim();
        let val: Value = after.parse().unwrap_or(1);
        Some((addr, val))
    }

    fn parse_ptx_load(&self, text: &str) -> Option<(String, String)> {
        // "ld.global.acquire.sys r0, [x]"
        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.len() >= 3 {
            let reg = parts[1].trim_matches(',').to_string();
            let bracket_start = text.find('[')? + 1;
            let bracket_end = text.find(']')?;
            let addr = text[bracket_start..bracket_end].trim().to_string();
            Some((reg, addr))
        } else {
            None
        }
    }

    /// Convert a symbolic address name (e.g. "x") to a numeric Address.
    fn addr_from_name(&self, name: &str) -> Address {
        // Simple deterministic mapping: first char as offset from base address.
        let base: Address = 0x100;
        let offset = name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        base + offset
    }

    /// Convert a symbolic register name (e.g. "r0") to a numeric RegId.
    fn reg_from_name(&self, name: &str) -> RegId {
        // Try parsing digits from the name.
        name.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<RegId>()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let input = r#"
test SB
thread 0
  W(x, 1)
  R(y) r0
thread 1
  W(y, 1)
  R(x) r1
forbidden: r0=0, r1=0
"#;
        let parser = LitmusParser::new();
        let test = parser.parse_simple(input).unwrap();
        assert_eq!(test.name, "SB");
        assert_eq!(test.num_threads(), 2);
        assert_eq!(test.num_events(), 4);
        assert_eq!(test.expected_outcomes.len(), 1);
        assert!(matches!(test.expected_outcomes[0].1, LitmusOutcome::Forbidden));
    }

    #[test]
    fn test_parse_ptx() {
        let input = r#"
Thread 0
  st.global.relaxed.sys [x], 1
  ld.global.acquire.sys r0, [y]
Thread 1
  st.global.relaxed.sys [y], 1
  ld.global.acquire.sys r1, [x]
"#;
        let parser = LitmusParser::new();
        let test = parser.parse_ptx(input).unwrap();
        assert_eq!(test.num_threads(), 2);
        assert_eq!(test.num_events(), 4);
    }

    #[test]
    fn test_parse_lisa() {
        let input = r#"
LISA
thread 0
  w[x] 1
  r[y] r0
thread 1
  w[y] 1
  r[x] r1
"#;
        let parser = LitmusParser::new();
        let test = parser.parse_lisa(input).unwrap();
        assert_eq!(test.num_threads(), 2);
        assert_eq!(test.num_events(), 4);
    }

    #[test]
    fn test_parse_herd() {
        let input = r#"
C SB
{ x=0; y=0; }
P0:
  W(x, 1);
  R(y) r0
P1:
  W(y, 1);
  R(x) r1
exists (r0=0, r1=0)
"#;
        let parser = LitmusParser::new();
        let test = parser.parse_herd(input).unwrap();
        assert_eq!(test.name, "SB");
        assert_eq!(test.num_threads(), 2);
    }

    #[test]
    fn test_auto_detect_simple() {
        let input = r#"
test MP
thread 0
  W(x, 1)
  W(y, 1)
thread 1
  R(y) r0
  R(x) r1
"#;
        let parser = LitmusParser::new();
        let test = parser.parse(input).unwrap();
        assert_eq!(test.name, "MP");
    }

    #[test]
    fn test_auto_detect_ptx() {
        let input = r#"
Thread 0
  st.global.relaxed.sys [x], 1
  ld.global.acquire.sys r0, [y]
"#;
        let parser = LitmusParser::new();
        let test = parser.parse(input).unwrap();
        assert!(test.num_threads() > 0);
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer = Tokenizer::new("W(x, 1); R(y) r0");
        let tokens = tokenizer.tokenize();
        assert!(tokens.iter().any(|t| matches!(t, Token::Ident(s) if s == "W")));
        assert!(tokens.iter().any(|t| matches!(t, Token::Number(1))));
    }

    #[test]
    fn test_parse_outcome() {
        let parser = LitmusParser::new();
        let outcome = parser.parse_outcome("r0=0, r1=0").unwrap();
        assert_eq!(outcome.memory.len(), 2);
        assert!(outcome.memory.values().all(|&v| v == 0));
    }

    #[test]
    fn test_parse_ptx_with_fence() {
        let input = r#"
Thread 0
  st.global.release.sys [x], 1
  membar.sys
  ld.global.acquire.sys r0, [y]
"#;
        let parser = LitmusParser::new();
        let test = parser.parse_ptx(input).unwrap();
        assert_eq!(test.threads[0].instructions.len(), 3);
    }

    #[test]
    fn test_empty_input() {
        let parser = LitmusParser::new();
        let result = parser.parse_simple("");
        // Should succeed with empty test
        assert!(result.is_ok());
    }

    #[test]
    fn test_thread_id_extraction() {
        let parser = LitmusParser::new();
        assert_eq!(parser.extract_thread_id("P0:"), 0);
        assert_eq!(parser.extract_thread_id("Thread 3"), 3);
        assert_eq!(parser.extract_thread_id("T1"), 1);
    }

    #[test]
    fn test_ptx_ordering_parse() {
        let parser = LitmusParser::new();
        assert_eq!(parser.parse_ptx_ordering("st.global.relaxed.sys"), Ordering::Relaxed);
        assert_eq!(parser.parse_ptx_ordering("ld.global.acquire.gpu"), Ordering::Acquire);
        assert_eq!(parser.parse_ptx_ordering("st.global.release.cta"), Ordering::Release);
    }
}
