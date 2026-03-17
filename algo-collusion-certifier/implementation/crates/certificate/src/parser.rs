//! Certificate parser.
//!
//! Tokenizer, recursive descent parser, and structural validator for
//! the CollusionProof certificate text format.

use crate::ast::*;
use crate::proof_term::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Parse error ──────────────────────────────────────────────────────────────

/// Error encountered during parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub context: Option<String>,
}

impl ParseError {
    pub fn new(message: &str, line: usize, column: usize) -> Self {
        Self {
            message: message.to_string(),
            line,
            column,
            context: None,
        }
    }

    pub fn with_context(mut self, ctx: &str) -> Self {
        self.context = Some(ctx.to_string());
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

// ── Token types ──────────────────────────────────────────────────────────────

/// Token produced by the certificate tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Token {
    // Keywords
    Certificate,
    Version,
    Timestamp,
    Scenario,
    Oracle,
    Alpha,
    Step,
    Data,
    Test,
    Equilibrium,
    Deviation,
    Punishment,
    CollusionPremium,
    Inference,
    Verdict,
    // Structural
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Arrow,
    Dot,
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    Eq,
    // Literals
    IntegerLit(i64),
    FloatLit(f64),
    StringLit(String),
    BoolLit(bool),
    // Identifiers
    Ident(String),
    // Special
    Eof,
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Token::Certificate
                | Token::Version
                | Token::Timestamp
                | Token::Scenario
                | Token::Oracle
                | Token::Alpha
                | Token::Step
                | Token::Data
                | Token::Test
                | Token::Equilibrium
                | Token::Deviation
                | Token::Punishment
                | Token::CollusionPremium
                | Token::Inference
                | Token::Verdict
        )
    }

    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Token::IntegerLit(_) | Token::FloatLit(_) | Token::StringLit(_) | Token::BoolLit(_)
        )
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Certificate => write!(f, "certificate"),
            Token::Version => write!(f, "version"),
            Token::Timestamp => write!(f, "timestamp"),
            Token::Scenario => write!(f, "scenario"),
            Token::Oracle => write!(f, "oracle"),
            Token::Alpha => write!(f, "alpha"),
            Token::Step => write!(f, "step"),
            Token::Data => write!(f, "data"),
            Token::Test => write!(f, "test"),
            Token::Equilibrium => write!(f, "equilibrium"),
            Token::Deviation => write!(f, "deviation"),
            Token::Punishment => write!(f, "punishment"),
            Token::CollusionPremium => write!(f, "collusion_premium"),
            Token::Inference => write!(f, "inference"),
            Token::Verdict => write!(f, "verdict"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Arrow => write!(f, "->"),
            Token::Dot => write!(f, "."),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::Lt => write!(f, "<"),
            Token::Le => write!(f, "<="),
            Token::Gt => write!(f, ">"),
            Token::Ge => write!(f, ">="),
            Token::EqEq => write!(f, "=="),
            Token::Ne => write!(f, "!="),
            Token::Eq => write!(f, "="),
            Token::IntegerLit(i) => write!(f, "{}", i),
            Token::FloatLit(v) => write!(f, "{}", v),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::BoolLit(b) => write!(f, "{}", b),
            Token::Ident(id) => write!(f, "{}", id),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

/// Tokenize certificate text into a stream of tokens.
pub struct CertificateTokenizer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl CertificateTokenizer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    /// Tokenize the entire input into a vector of tokens.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            if tok == Token::Eof {
                tokens.push(tok);
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Ok(Token::Eof);
        }

        let ch = self.input[self.pos];

        // String literal
        if ch == '"' {
            return self.read_string();
        }

        // Number literal
        if ch.is_ascii_digit() || (ch == '-' && self.peek_is_digit()) {
            return self.read_number();
        }

        // Identifier or keyword
        if ch.is_alphabetic() || ch == '_' {
            return self.read_identifier();
        }

        // Two-character operators
        if self.pos + 1 < self.input.len() {
            let two = format!("{}{}", ch, self.input[self.pos + 1]);
            let tok = match two.as_str() {
                "<=" => Some(Token::Le),
                ">=" => Some(Token::Ge),
                "==" => Some(Token::EqEq),
                "!=" => Some(Token::Ne),
                "->" => Some(Token::Arrow),
                _ => None,
            };
            if let Some(t) = tok {
                self.advance();
                self.advance();
                return Ok(t);
            }
        }

        // Single-character tokens
        let tok = match ch {
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            ',' => Token::Comma,
            ':' => Token::Colon,
            ';' => Token::Semicolon,
            '.' => Token::Dot,
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '^' => Token::Caret,
            '<' => Token::Lt,
            '>' => Token::Gt,
            '=' => Token::Eq,
            _ => {
                return Err(ParseError::new(
                    &format!("Unexpected character: '{}'", ch),
                    self.line,
                    self.col,
                ));
            }
        };
        self.advance();
        Ok(tok)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch.is_whitespace() {
                if ch == '\n' {
                    self.line += 1;
                    self.col = 1;
                } else {
                    self.col += 1;
                }
                self.pos += 1;
                continue;
            }
            // Line comment
            if ch == '/' && self.pos + 1 < self.input.len() && self.input[self.pos + 1] == '/' {
                while self.pos < self.input.len() && self.input[self.pos] != '\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn advance(&mut self) {
        if self.pos < self.input.len() {
            if self.input[self.pos] == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn peek_is_digit(&self) -> bool {
        self.pos + 1 < self.input.len() && self.input[self.pos + 1].is_ascii_digit()
    }

    fn read_string(&mut self) -> Result<Token, ParseError> {
        self.advance(); // skip opening quote
        let mut s = String::new();
        while self.pos < self.input.len() && self.input[self.pos] != '"' {
            if self.input[self.pos] == '\\' && self.pos + 1 < self.input.len() {
                self.advance();
                match self.input[self.pos] {
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    _ => s.push(self.input[self.pos]),
                }
            } else {
                s.push(self.input[self.pos]);
            }
            self.advance();
        }
        if self.pos >= self.input.len() {
            return Err(ParseError::new("Unterminated string literal", self.line, self.col));
        }
        self.advance(); // skip closing quote
        Ok(Token::StringLit(s))
    }

    fn read_number(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        let negative = self.input[self.pos] == '-';
        if negative {
            self.advance();
        }

        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.advance();
        }

        let is_float = self.pos < self.input.len() && self.input[self.pos] == '.';
        if is_float {
            self.advance();
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.advance();
            }
        }

        // Scientific notation
        if self.pos < self.input.len()
            && (self.input[self.pos] == 'e' || self.input[self.pos] == 'E')
        {
            self.advance();
            if self.pos < self.input.len()
                && (self.input[self.pos] == '+' || self.input[self.pos] == '-')
            {
                self.advance();
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.advance();
            }
        }

        let num_str: String = self.input[start..self.pos].iter().collect();

        if is_float || num_str.contains('e') || num_str.contains('E') {
            let val: f64 = num_str
                .parse()
                .map_err(|_| ParseError::new(&format!("Invalid float: {}", num_str), self.line, self.col))?;
            Ok(Token::FloatLit(val))
        } else {
            let val: i64 = num_str
                .parse()
                .map_err(|_| ParseError::new(&format!("Invalid integer: {}", num_str), self.line, self.col))?;
            Ok(Token::IntegerLit(val))
        }
    }

    fn read_identifier(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        while self.pos < self.input.len()
            && (self.input[self.pos].is_alphanumeric() || self.input[self.pos] == '_')
        {
            self.advance();
        }

        let word: String = self.input[start..self.pos].iter().collect();

        let tok = match word.as_str() {
            "certificate" => Token::Certificate,
            "version" => Token::Version,
            "timestamp" => Token::Timestamp,
            "scenario" => Token::Scenario,
            "oracle" => Token::Oracle,
            "alpha" => Token::Alpha,
            "step" => Token::Step,
            "data" => Token::Data,
            "test" => Token::Test,
            "equilibrium" => Token::Equilibrium,
            "deviation" => Token::Deviation,
            "punishment" => Token::Punishment,
            "collusion_premium" => Token::CollusionPremium,
            "inference" => Token::Inference,
            "verdict" => Token::Verdict,
            "true" => Token::BoolLit(true),
            "false" => Token::BoolLit(false),
            _ => Token::Ident(word),
        };
        Ok(tok)
    }
}

// ── Recursive descent parser ─────────────────────────────────────────────────

/// Recursive descent parser for the certificate text format.
pub struct RecursiveDescentParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl RecursiveDescentParser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<Token, ParseError> {
        let tok = self.advance();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            Ok(tok)
        } else {
            Err(ParseError::new(
                &format!("Expected {}, got {}", expected, tok),
                0,
                self.pos,
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            tok => Err(ParseError::new(
                &format!("Expected identifier, got {}", tok),
                0,
                self.pos,
            )),
        }
    }

    fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.advance() {
            Token::StringLit(s) => Ok(s),
            tok => Err(ParseError::new(
                &format!("Expected string literal, got {}", tok),
                0,
                self.pos,
            )),
        }
    }

    fn expect_float(&mut self) -> Result<f64, ParseError> {
        match self.advance() {
            Token::FloatLit(v) => Ok(v),
            Token::IntegerLit(i) => Ok(i as f64),
            tok => Err(ParseError::new(
                &format!("Expected number, got {}", tok),
                0,
                self.pos,
            )),
        }
    }

    fn expect_integer(&mut self) -> Result<i64, ParseError> {
        match self.advance() {
            Token::IntegerLit(i) => Ok(i),
            tok => Err(ParseError::new(
                &format!("Expected integer, got {}", tok),
                0,
                self.pos,
            )),
        }
    }

    /// Parse a complete certificate.
    pub fn parse_certificate(&mut self) -> Result<CertificateAST, ParseError> {
        self.expect(&Token::Certificate)?;
        self.expect(&Token::LBrace)?;

        let header = self.parse_header()?;
        let body = self.parse_body()?;

        self.expect(&Token::RBrace)?;

        Ok(CertificateAST::new(header, body))
    }

    fn parse_header(&mut self) -> Result<CertificateHeader, ParseError> {
        let mut version = "1.0.0".to_string();
        let mut timestamp = String::new();
        let mut scenario = String::new();
        let mut oracle_str = "Layer0".to_string();
        let mut alpha = 0.05;

        while !matches!(self.peek(), Token::Step | Token::RBrace | Token::Eof) {
            match self.peek().clone() {
                Token::Version => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    version = self.expect_string()?;
                }
                Token::Timestamp => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    timestamp = self.expect_string()?;
                }
                Token::Scenario => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    scenario = self.expect_string()?;
                }
                Token::Oracle => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    oracle_str = self.expect_ident()?;
                }
                Token::Alpha => {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    alpha = self.expect_float()?;
                }
                _ => {
                    // Skip unknown header fields
                    self.advance();
                }
            }
            // Skip optional semicolons
            if matches!(self.peek(), Token::Semicolon) {
                self.advance();
            }
        }

        let oracle_level = match oracle_str.as_str() {
            "Layer0" | "layer0" | "L0" => shared_types::OracleAccessLevel::Layer0,
            "Layer1" | "layer1" | "L1" => shared_types::OracleAccessLevel::Layer1,
            "Layer2" | "layer2" | "L2" => shared_types::OracleAccessLevel::Layer2,
            _ => {
                return Err(ParseError::new(
                    &format!("Unknown oracle level: {}", oracle_str),
                    0,
                    self.pos,
                ))
            }
        };

        Ok(CertificateHeader {
            version,
            timestamp,
            scenario,
            oracle_level,
            alpha: shared_types::SignificanceLevel::new(alpha)
                .map_err(|e| ParseError::new(&e, 0, self.pos))?,
        })
    }

    fn parse_body(&mut self) -> Result<CertificateBody, ParseError> {
        let mut body = CertificateBody::new();
        while matches!(self.peek(), Token::Step) {
            let step = self.parse_step()?;
            body.push(step);
        }
        Ok(body)
    }

    fn parse_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::Step)?;
        let step_type = self.expect_ident()?;

        match step_type.as_str() {
            "data" | "DataDeclaration" => self.parse_data_step(),
            "test" | "StatisticalTest" => self.parse_test_step(),
            "equilibrium" | "EquilibriumClaim" => self.parse_equilibrium_step(),
            "deviation" | "DeviationBound" => self.parse_deviation_step(),
            "punishment" | "PunishmentEvidence" => self.parse_punishment_step(),
            "cp" | "CollusionPremium" => self.parse_cp_step(),
            "inference" | "Inference" => self.parse_inference_step(),
            "verdict" | "Verdict" => self.parse_verdict_step(),
            _ => Err(ParseError::new(
                &format!("Unknown step type: {}", step_type),
                0,
                self.pos,
            )),
        }
    }

    fn parse_data_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let seg_type = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let start = self.expect_integer()? as usize;
        self.expect(&Token::Comma)?;
        let end = self.expect_integer()? as usize;
        self.expect(&Token::Comma)?;
        let hash = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let num_players = self.expect_integer()? as usize;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::DataDeclaration(
            TrajectoryRef::new(&ref_id),
            SegmentSpec::new(&seg_type, start, end, &hash, num_players),
        ))
    }

    fn parse_test_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let test_name = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let category = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let stat = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let pval = self.expect_float()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::StatisticalTest(
            TestRef::new(&ref_id),
            TestType::new(&test_name, &category),
            Statistic::new(stat),
            PValueWrapper::new(pval),
        ))
    }

    fn parse_equilibrium_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let num_players = self.expect_integer()? as usize;
        self.expect(&Token::Comma)?;
        let market_type = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let prices = self.parse_float_list()?;
        self.expect(&Token::Comma)?;
        let profits = self.parse_float_list()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::EquilibriumClaim(
            EquilibriumRef::new(&ref_id),
            GameSpec::new(num_players, &market_type),
            NashProfile::new(prices, profits),
        ))
    }

    fn parse_deviation_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let player = self.expect_integer()? as usize;
        self.expect(&Token::Comma)?;
        let bound = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let conf = self.expect_float()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::DeviationBound(
            DeviationRef::new(&ref_id),
            shared_types::PlayerId(player),
            Bound::upper(bound),
            shared_types::ConfidenceLevel::new(conf)
                .map_err(|e| ParseError::new(&e, 0, self.pos))?,
        ))
    }

    fn parse_punishment_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let player = self.expect_integer()? as usize;
        self.expect(&Token::Comma)?;
        let drop_val = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let pval = self.expect_float()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::PunishmentEvidence(
            PunishmentRef::new(&ref_id),
            shared_types::PlayerId(player),
            PayoffDrop::new(drop_val, 0.0),
            PValueWrapper::new(pval),
        ))
    }

    fn parse_cp_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let value = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let lo = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let hi = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let level = self.expect_float()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::CollusionPremium(
            CPRef::new(&ref_id),
            Value::new(value),
            CIWrapper::new(lo, hi, level),
        ))
    }

    fn parse_inference_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let ref_id = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let rule_name = self.expect_string()?;
        self.expect(&Token::Comma)?;
        let premises = self.parse_string_list()?;
        self.expect(&Token::Comma)?;
        let conclusion = self.expect_string()?;
        self.expect(&Token::RParen)?;

        Ok(ProofStep::Inference(
            InferenceRef::new(&ref_id),
            Rule::new(&rule_name),
            Premises::new(premises),
            Conclusion::new(&conclusion),
        ))
    }

    fn parse_verdict_step(&mut self) -> Result<ProofStep, ParseError> {
        self.expect(&Token::LParen)?;
        let verdict_str = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let confidence = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let refs = self.parse_string_list()?;
        self.expect(&Token::RParen)?;

        let verdict = match verdict_str.as_str() {
            "Collusive" | "collusive" => VerdictType::Collusive,
            "Competitive" | "competitive" => VerdictType::Competitive,
            "Inconclusive" | "inconclusive" => VerdictType::Inconclusive,
            _ => {
                return Err(ParseError::new(
                    &format!("Unknown verdict type: {}", verdict_str),
                    0,
                    self.pos,
                ))
            }
        };

        Ok(ProofStep::Verdict(
            verdict,
            Confidence::new(confidence),
            SupportingRefs::new(refs),
        ))
    }

    fn parse_float_list(&mut self) -> Result<Vec<f64>, ParseError> {
        self.expect(&Token::LBracket)?;
        let mut values = Vec::new();
        if !matches!(self.peek(), Token::RBracket) {
            values.push(self.expect_float()?);
            while matches!(self.peek(), Token::Comma) {
                self.advance();
                if matches!(self.peek(), Token::RBracket) {
                    break;
                }
                values.push(self.expect_float()?);
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(values)
    }

    fn parse_string_list(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect(&Token::LBracket)?;
        let mut values = Vec::new();
        if !matches!(self.peek(), Token::RBracket) {
            values.push(self.expect_string()?);
            while matches!(self.peek(), Token::Comma) {
                self.advance();
                if matches!(self.peek(), Token::RBracket) {
                    break;
                }
                values.push(self.expect_string()?);
            }
        }
        self.expect(&Token::RBracket)?;
        Ok(values)
    }
}

// ── Parse expression ─────────────────────────────────────────────────────────

/// Parse an expression from a token stream.
pub fn parse_expression(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    parse_comparison(tokens, pos)
}

fn parse_comparison(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    let left = parse_additive(tokens, pos)?;

    let op = match tokens.get(*pos) {
        Some(Token::Lt) => Some(ComparisonOp::Lt),
        Some(Token::Le) => Some(ComparisonOp::Le),
        Some(Token::Gt) => Some(ComparisonOp::Gt),
        Some(Token::Ge) => Some(ComparisonOp::Ge),
        Some(Token::EqEq) => Some(ComparisonOp::Eq),
        Some(Token::Ne) => Some(ComparisonOp::Ne),
        _ => None,
    };

    if let Some(op) = op {
        *pos += 1;
        let right = parse_additive(tokens, pos)?;
        Ok(Expression::Comparison {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    } else {
        Ok(left)
    }
}

fn parse_additive(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    let mut left = parse_multiplicative(tokens, pos)?;

    while let Some(tok) = tokens.get(*pos) {
        let op = match tok {
            Token::Plus => BinaryOp::Add,
            Token::Minus => BinaryOp::Sub,
            _ => break,
        };
        *pos += 1;
        let right = parse_multiplicative(tokens, pos)?;
        left = Expression::BinaryExpr {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_multiplicative(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    let mut left = parse_unary(tokens, pos)?;

    while let Some(tok) = tokens.get(*pos) {
        let op = match tok {
            Token::Star => BinaryOp::Mul,
            Token::Slash => BinaryOp::Div,
            Token::Caret => BinaryOp::Pow,
            _ => break,
        };
        *pos += 1;
        let right = parse_unary(tokens, pos)?;
        left = Expression::BinaryExpr {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    if let Some(Token::Minus) = tokens.get(*pos) {
        *pos += 1;
        let operand = parse_primary(tokens, pos)?;
        return Ok(Expression::UnaryExpr {
            op: UnaryOp::Neg,
            operand: Box::new(operand),
        });
    }
    parse_primary(tokens, pos)
}

fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
    match tokens.get(*pos) {
        Some(Token::FloatLit(v)) => {
            let val = *v;
            *pos += 1;
            Ok(Expression::float(val))
        }
        Some(Token::IntegerLit(i)) => {
            let val = *i;
            *pos += 1;
            Ok(Expression::Literal(LiteralValue::Integer(val)))
        }
        Some(Token::BoolLit(b)) => {
            let val = *b;
            *pos += 1;
            Ok(Expression::boolean(val))
        }
        Some(Token::StringLit(s)) => {
            let val = s.clone();
            *pos += 1;
            Ok(Expression::Literal(LiteralValue::String(val)))
        }
        Some(Token::Ident(name)) => {
            let name = name.clone();
            *pos += 1;
            Ok(Expression::var(&name))
        }
        Some(Token::LParen) => {
            *pos += 1;
            let expr = parse_expression(tokens, pos)?;
            if !matches!(tokens.get(*pos), Some(Token::RParen)) {
                return Err(ParseError::new("Expected ')'", 0, *pos));
            }
            *pos += 1;
            Ok(expr)
        }
        Some(Token::LBracket) => {
            *pos += 1;
            let lo = parse_expression(tokens, pos)?;
            if !matches!(tokens.get(*pos), Some(Token::Comma)) {
                return Err(ParseError::new("Expected ',' in interval", 0, *pos));
            }
            *pos += 1;
            let hi = parse_expression(tokens, pos)?;
            if !matches!(tokens.get(*pos), Some(Token::RBracket)) {
                return Err(ParseError::new("Expected ']'", 0, *pos));
            }
            *pos += 1;
            Ok(Expression::IntervalExpr(Box::new(lo), Box::new(hi)))
        }
        _ => Err(ParseError::new(
            &format!(
                "Unexpected token: {:?}",
                tokens.get(*pos).unwrap_or(&Token::Eof)
            ),
            0,
            *pos,
        )),
    }
}

// ── Top-level parse function ─────────────────────────────────────────────────

/// Parse a certificate from text input.
pub fn parse_certificate(input: &str) -> Result<CertificateAST, ParseError> {
    let mut tokenizer = CertificateTokenizer::new(input);
    let tokens = tokenizer.tokenize()?;
    let mut parser = RecursiveDescentParser::new(tokens);
    parser.parse_certificate()
}

/// Validate structural well-formedness of a parsed AST.
pub fn validate_ast(cert: &CertificateAST) -> Result<(), ParseError> {
    // Check version
    if cert.header.version.is_empty() {
        return Err(ParseError::new("Empty certificate version", 0, 0));
    }

    // Check alpha
    let alpha = cert.header.alpha.value();
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(ParseError::new(
            &format!("Invalid alpha: {}", alpha),
            0,
            0,
        ));
    }

    // Check for duplicate references
    let mut refs = std::collections::HashSet::new();
    for step in &cert.body.steps {
        if let Some(r) = step.declared_ref() {
            if !refs.insert(r.clone()) {
                return Err(ParseError::new(
                    &format!("Duplicate reference: {}", r),
                    0,
                    0,
                ));
            }
        }
    }

    // Check that all dependency refs exist
    for step in &cert.body.steps {
        for dep in step.dependency_refs() {
            if !refs.contains(&dep) {
                return Err(ParseError::new(
                    &format!("Undeclared dependency reference: {}", dep),
                    0,
                    0,
                ));
            }
        }
    }

    Ok(())
}

// ── Parse proof term ─────────────────────────────────────────────────────────

/// Parse a proof term from a token stream.
pub fn parse_proof_term(tokens: &[Token], pos: &mut usize) -> Result<ProofTerm, ParseError> {
    match tokens.get(*pos) {
        Some(Token::Ident(name)) if name == "Axiom" => {
            *pos += 1;
            if !matches!(tokens.get(*pos), Some(Token::LParen)) {
                return Err(ParseError::new("Expected '(' after Axiom", 0, *pos));
            }
            *pos += 1;
            let schema_name = match tokens.get(*pos) {
                Some(Token::Ident(s)) => s.clone(),
                _ => return Err(ParseError::new("Expected axiom schema name", 0, *pos)),
            };
            *pos += 1;
            if !matches!(tokens.get(*pos), Some(Token::RParen)) {
                return Err(ParseError::new("Expected ')'", 0, *pos));
            }
            *pos += 1;

            let schema = match schema_name.as_str() {
                "CompetitiveNullDef" => AxiomSchema::CompetitiveNullDef,
                "TestSoundness" => AxiomSchema::TestSoundness,
                "CorrelationBound" => AxiomSchema::CorrelationBound,
                "NashEquilibriumDef" => AxiomSchema::NashEquilibriumDef,
                _ => AxiomSchema::CompetitiveNullDef,
            };
            Ok(ProofTerm::Axiom(schema, Instantiation::new()))
        }
        Some(Token::Ident(name)) if name == "Ref" => {
            *pos += 1;
            if !matches!(tokens.get(*pos), Some(Token::LParen)) {
                return Err(ParseError::new("Expected '('", 0, *pos));
            }
            *pos += 1;
            let ref_name = match tokens.get(*pos) {
                Some(Token::Ident(s)) => s.clone(),
                Some(Token::StringLit(s)) => s.clone(),
                _ => return Err(ParseError::new("Expected reference name", 0, *pos)),
            };
            *pos += 1;
            if !matches!(tokens.get(*pos), Some(Token::RParen)) {
                return Err(ParseError::new("Expected ')'", 0, *pos));
            }
            *pos += 1;
            Ok(ProofTerm::Reference(ref_name))
        }
        _ => Err(ParseError::new(
            &format!(
                "Unexpected token in proof term: {:?}",
                tokens.get(*pos).unwrap_or(&Token::Eof)
            ),
            0,
            *pos,
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let mut tok = CertificateTokenizer::new("certificate { }");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Certificate);
        assert_eq!(tokens[1], Token::LBrace);
        assert_eq!(tokens[2], Token::RBrace);
        assert_eq!(tokens[3], Token::Eof);
    }

    #[test]
    fn test_tokenizer_string_literal() {
        let mut tok = CertificateTokenizer::new(r#""hello world""#);
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::StringLit("hello world".to_string()));
    }

    #[test]
    fn test_tokenizer_numbers() {
        let mut tok = CertificateTokenizer::new("42 3.14 -5 1e-3");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::IntegerLit(42));
        assert_eq!(tokens[1], Token::FloatLit(3.14));
        assert_eq!(tokens[2], Token::Minus);
        assert_eq!(tokens[3], Token::IntegerLit(5));
    }

    #[test]
    fn test_tokenizer_keywords() {
        let mut tok = CertificateTokenizer::new("version scenario oracle alpha verdict");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Version);
        assert_eq!(tokens[1], Token::Scenario);
        assert_eq!(tokens[2], Token::Oracle);
        assert_eq!(tokens[3], Token::Alpha);
        assert_eq!(tokens[4], Token::Verdict);
    }

    #[test]
    fn test_tokenizer_operators() {
        let mut tok = CertificateTokenizer::new("+ - * / < <= > >= == != ->");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Plus);
        assert_eq!(tokens[1], Token::Minus);
        assert_eq!(tokens[2], Token::Star);
        assert_eq!(tokens[3], Token::Slash);
        assert_eq!(tokens[4], Token::Lt);
        assert_eq!(tokens[5], Token::Le);
        assert_eq!(tokens[6], Token::Gt);
        assert_eq!(tokens[7], Token::Ge);
        assert_eq!(tokens[8], Token::EqEq);
        assert_eq!(tokens[9], Token::Ne);
        assert_eq!(tokens[10], Token::Arrow);
    }

    #[test]
    fn test_tokenizer_comments() {
        let mut tok = CertificateTokenizer::new("alpha // this is a comment\nversion");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Alpha);
        assert_eq!(tokens[1], Token::Version);
    }

    #[test]
    fn test_tokenizer_boolean() {
        let mut tok = CertificateTokenizer::new("true false");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::BoolLit(true));
        assert_eq!(tokens[1], Token::BoolLit(false));
    }

    #[test]
    fn test_parse_simple_certificate() {
        let input = r#"certificate {
            version: "1.0.0";
            scenario: "test";
            oracle: Layer0;
            alpha: 0.05;
            step verdict(Competitive, 0.5, [])
        }"#;
        let result = parse_certificate(input);
        assert!(result.is_ok());
        let cert = result.unwrap();
        assert_eq!(cert.header.scenario, "test");
        assert_eq!(cert.step_count(), 1);
    }

    #[test]
    fn test_parse_certificate_with_data_step() {
        let input = r#"certificate {
            version: "1.0.0";
            scenario: "s1";
            oracle: Layer0;
            alpha: 0.05;
            step DataDeclaration(traj_0, "testing", 0, 500, "hash123", 2)
            step verdict(Competitive, 0.5, [])
        }"#;
        let result = parse_certificate(input);
        assert!(result.is_ok());
        let cert = result.unwrap();
        assert_eq!(cert.step_count(), 2);
    }

    #[test]
    fn test_parse_certificate_with_test_step() {
        let input = r#"certificate {
            version: "1.0.0";
            scenario: "s1";
            oracle: Layer0;
            alpha: 0.05;
            step StatisticalTest(test_0, "PriceCorrelation", "layer0", 3.5, 0.001)
            step verdict(Competitive, 0.5, [])
        }"#;
        let result = parse_certificate(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_expression_simple() {
        let mut tok = CertificateTokenizer::new("3.14 + 2.0");
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let expr = parse_expression(&tokens, &mut pos).unwrap();
        let val = expr.try_eval_f64().unwrap();
        assert!((val - 5.14).abs() < 1e-12);
    }

    #[test]
    fn test_parse_expression_comparison() {
        let mut tok = CertificateTokenizer::new("1.0 < 2.0");
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let expr = parse_expression(&tokens, &mut pos).unwrap();
        assert!(matches!(expr, Expression::Comparison { .. }));
    }

    #[test]
    fn test_parse_expression_multiplication() {
        let mut tok = CertificateTokenizer::new("3.0 * 4.0 + 1.0");
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let expr = parse_expression(&tokens, &mut pos).unwrap();
        let val = expr.try_eval_f64().unwrap();
        assert!((val - 13.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_expression_parenthesized() {
        let mut tok = CertificateTokenizer::new("(2.0 + 3.0) * 4.0");
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let expr = parse_expression(&tokens, &mut pos).unwrap();
        let val = expr.try_eval_f64().unwrap();
        assert!((val - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_proof_term_reference() {
        let mut tok = CertificateTokenizer::new(r#"Ref("test_0")"#);
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let term = parse_proof_term(&tokens, &mut pos).unwrap();
        assert!(matches!(term, ProofTerm::Reference(r) if r == "test_0"));
    }

    #[test]
    fn test_parse_proof_term_axiom() {
        let mut tok = CertificateTokenizer::new("Axiom(TestSoundness)");
        let tokens = tok.tokenize().unwrap();
        let mut pos = 0;
        let term = parse_proof_term(&tokens, &mut pos).unwrap();
        assert!(matches!(
            term,
            ProofTerm::Axiom(AxiomSchema::TestSoundness, _)
        ));
    }

    #[test]
    fn test_validate_ast_valid() {
        let header = CertificateHeader::new("s", shared_types::OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        assert!(validate_ast(&cert).is_ok());
    }

    #[test]
    fn test_validate_ast_duplicate_ref() {
        let header = CertificateHeader::new("s", shared_types::OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("testing", 0, 100, "h", 2),
        ));
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_0"),
            SegmentSpec::new("training", 100, 200, "h", 2),
        ));
        let cert = CertificateAST::new(header, body);
        assert!(validate_ast(&cert).is_err());
    }

    #[test]
    fn test_validate_ast_undeclared_dep() {
        let header = CertificateHeader::new("s", shared_types::OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec!["nonexistent".into()]),
        ));
        let cert = CertificateAST::new(header, body);
        assert!(validate_ast(&cert).is_err());
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::new("unexpected token", 5, 10);
        let s = format!("{}", err);
        assert!(s.contains("5:10"));
        assert!(s.contains("unexpected token"));
    }

    #[test]
    fn test_token_classification() {
        assert!(Token::Certificate.is_keyword());
        assert!(!Token::FloatLit(1.0).is_keyword());
        assert!(Token::FloatLit(1.0).is_literal());
        assert!(!Token::Plus.is_literal());
    }

    #[test]
    fn test_tokenizer_escape_string() {
        let mut tok = CertificateTokenizer::new(r#""hello\nworld""#);
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::StringLit("hello\nworld".to_string()));
    }
}
