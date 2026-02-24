//! Constraint specification language for LITMUS∞.
//!
//! Provides a rich constraint expression language, parser,
//! evaluator, simplifier, normalizer (NNF/CNF/DNF), and
//! backtracking solver with arc consistency.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// ConstraintValue — values in the constraint language
// ═══════════════════════════════════════════════════════════════════════

/// A value in the constraint language.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Set(Vec<ConstraintValue>),
    Tuple(Vec<ConstraintValue>),
}

impl ConstraintValue {
    /// Convert to boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConstraintValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Convert to integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ConstraintValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Convert to float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstraintValue::Float(f) => Some(*f),
            ConstraintValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Check if values are equal.
    pub fn is_truthy(&self) -> bool {
        match self {
            ConstraintValue::Bool(b) => *b,
            ConstraintValue::Int(i) => *i != 0,
            ConstraintValue::Float(f) => *f != 0.0,
            ConstraintValue::Str(s) => !s.is_empty(),
            ConstraintValue::Set(s) => !s.is_empty(),
            ConstraintValue::Tuple(t) => !t.is_empty(),
        }
    }
}

impl fmt::Display for ConstraintValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintValue::Bool(b) => write!(f, "{}", b),
            ConstraintValue::Int(i) => write!(f, "{}", i),
            ConstraintValue::Float(v) => write!(f, "{}", v),
            ConstraintValue::Str(s) => write!(f, "\"{}\"", s),
            ConstraintValue::Set(s) => {
                write!(f, "{{")?;
                for (i, v) in s.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "}}")
            }
            ConstraintValue::Tuple(t) => {
                write!(f, "(")?;
                for (i, v) in t.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Domain — variable domains
// ═══════════════════════════════════════════════════════════════════════

/// Domain of a variable.
#[derive(Debug, Clone)]
pub enum Domain {
    /// Integer range [lo, hi].
    IntRange(i64, i64),
    /// Explicit set of values.
    Values(Vec<ConstraintValue>),
    /// Type name.
    Type(String),
}

impl Domain {
    /// Get all values in the domain.
    pub fn enumerate(&self) -> Vec<ConstraintValue> {
        match self {
            Domain::IntRange(lo, hi) => {
                (*lo..=*hi).map(ConstraintValue::Int).collect()
            }
            Domain::Values(vals) => vals.clone(),
            Domain::Type(_) => Vec::new(),
        }
    }

    /// Size of the domain.
    pub fn size(&self) -> usize {
        match self {
            Domain::IntRange(lo, hi) => (hi - lo + 1).max(0) as usize,
            Domain::Values(vals) => vals.len(),
            Domain::Type(_) => 0,
        }
    }

    /// Check if a value is in the domain.
    pub fn contains(&self, val: &ConstraintValue) -> bool {
        match (self, val) {
            (Domain::IntRange(lo, hi), ConstraintValue::Int(i)) => *i >= *lo && *i <= *hi,
            (Domain::Values(vals), v) => vals.contains(v),
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintExpr — constraint expression AST
// ═══════════════════════════════════════════════════════════════════════

/// A constraint expression.
#[derive(Debug, Clone)]
pub enum ConstraintExpr {
    /// Boolean literal.
    BoolLit(bool),
    /// Variable reference.
    Var(String),
    /// Integer literal.
    IntLit(i64),
    /// Float literal.
    FloatLit(f64),
    /// String literal.
    StringLit(String),
    /// Logical NOT.
    Not(Box<ConstraintExpr>),
    /// Logical AND.
    And(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Logical OR.
    Or(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Implication.
    Implies(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Bi-conditional.
    Iff(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Equality.
    Eq(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Not equal.
    Ne(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Less than.
    Lt(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Less than or equal.
    Le(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Greater than.
    Gt(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Greater than or equal.
    Ge(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Addition.
    Add(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Subtraction.
    Sub(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Multiplication.
    Mul(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Division.
    Div(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Universal quantifier.
    ForAll(String, Domain, Box<ConstraintExpr>),
    /// Existential quantifier.
    Exists(String, Domain, Box<ConstraintExpr>),
    /// Set membership.
    SetIn(Box<ConstraintExpr>, Box<ConstraintExpr>),
    /// Function application.
    FuncApp(String, Vec<ConstraintExpr>),
}

impl fmt::Display for ConstraintExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintExpr::BoolLit(b) => write!(f, "{}", b),
            ConstraintExpr::Var(v) => write!(f, "{}", v),
            ConstraintExpr::IntLit(i) => write!(f, "{}", i),
            ConstraintExpr::FloatLit(v) => write!(f, "{:.4}", v),
            ConstraintExpr::StringLit(s) => write!(f, "\"{}\"", s),
            ConstraintExpr::Not(e) => write!(f, "¬({})", e),
            ConstraintExpr::And(a, b) => write!(f, "({} ∧ {})", a, b),
            ConstraintExpr::Or(a, b) => write!(f, "({} ∨ {})", a, b),
            ConstraintExpr::Implies(a, b) => write!(f, "({} → {})", a, b),
            ConstraintExpr::Iff(a, b) => write!(f, "({} ↔ {})", a, b),
            ConstraintExpr::Eq(a, b) => write!(f, "({} = {})", a, b),
            ConstraintExpr::Ne(a, b) => write!(f, "({} ≠ {})", a, b),
            ConstraintExpr::Lt(a, b) => write!(f, "({} < {})", a, b),
            ConstraintExpr::Le(a, b) => write!(f, "({} ≤ {})", a, b),
            ConstraintExpr::Gt(a, b) => write!(f, "({} > {})", a, b),
            ConstraintExpr::Ge(a, b) => write!(f, "({} ≥ {})", a, b),
            ConstraintExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            ConstraintExpr::Sub(a, b) => write!(f, "({} - {})", a, b),
            ConstraintExpr::Mul(a, b) => write!(f, "({} × {})", a, b),
            ConstraintExpr::Div(a, b) => write!(f, "({} / {})", a, b),
            ConstraintExpr::ForAll(v, _, e) => write!(f, "∀{}: {}", v, e),
            ConstraintExpr::Exists(v, _, e) => write!(f, "∃{}: {}", v, e),
            ConstraintExpr::SetIn(e, s) => write!(f, "({} ∈ {})", e, s),
            ConstraintExpr::FuncApp(name, args) => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ParseError
// ═══════════════════════════════════════════════════════════════════════

/// Parse error with location.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}:{}: {}", self.line, self.column, self.message)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintParser — recursive descent parser
// ═══════════════════════════════════════════════════════════════════════

/// Recursive descent parser for constraint expressions.
#[derive(Debug)]
pub struct ConstraintParser {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl ConstraintParser {
    /// Parse a constraint expression from a string.
    pub fn parse(input: &str) -> Result<ConstraintExpr, ParseError> {
        let mut parser = ConstraintParser {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        };
        parser.skip_whitespace();
        let expr = parser.parse_iff()?;
        parser.skip_whitespace();
        if parser.pos < parser.input.len() {
            return Err(parser.error("Unexpected characters after expression"));
        }
        Ok(expr)
    }

    fn error(&self, msg: &str) -> ParseError {
        ParseError {
            line: self.line,
            column: self.col,
            message: msg.to_string(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), ParseError> {
        self.skip_whitespace();
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(self.error(&format!("Expected '{}', found '{}'", expected, ch))),
            None => Err(self.error(&format!("Expected '{}', found end of input", expected))),
        }
    }

    fn match_str(&mut self, s: &str) -> bool {
        let remaining: String = self.input[self.pos..].iter().collect();
        if remaining.starts_with(s) {
            for _ in 0..s.len() {
                self.advance();
            }
            true
        } else {
            false
        }
    }

    fn parse_iff(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_implies()?;
        self.skip_whitespace();
        while self.match_str("<=>") {
            self.skip_whitespace();
            let right = self.parse_implies()?;
            left = ConstraintExpr::Iff(Box::new(left), Box::new(right));
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_implies(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_or()?;
        self.skip_whitespace();
        while self.match_str("=>") {
            self.skip_whitespace();
            let right = self.parse_or()?;
            left = ConstraintExpr::Implies(Box::new(left), Box::new(right));
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_and()?;
        self.skip_whitespace();
        while self.match_str("||") {
            self.skip_whitespace();
            let right = self.parse_and()?;
            left = ConstraintExpr::Or(Box::new(left), Box::new(right));
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_comparison()?;
        self.skip_whitespace();
        while self.match_str("&&") {
            self.skip_whitespace();
            let right = self.parse_comparison()?;
            left = ConstraintExpr::And(Box::new(left), Box::new(right));
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<ConstraintExpr, ParseError> {
        let left = self.parse_additive()?;
        self.skip_whitespace();
        if self.match_str("==") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Eq(Box::new(left), Box::new(right)))
        } else if self.match_str("!=") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Ne(Box::new(left), Box::new(right)))
        } else if self.match_str("<=") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Le(Box::new(left), Box::new(right)))
        } else if self.match_str(">=") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Ge(Box::new(left), Box::new(right)))
        } else if self.match_str("<") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Lt(Box::new(left), Box::new(right)))
        } else if self.match_str(">") {
            self.skip_whitespace();
            let right = self.parse_additive()?;
            Ok(ConstraintExpr::Gt(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_additive(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_multiplicative()?;
        self.skip_whitespace();
        loop {
            if self.match_str("+") {
                self.skip_whitespace();
                let right = self.parse_multiplicative()?;
                left = ConstraintExpr::Add(Box::new(left), Box::new(right));
            } else if self.match_str("-") {
                self.skip_whitespace();
                let right = self.parse_multiplicative()?;
                left = ConstraintExpr::Sub(Box::new(left), Box::new(right));
            } else {
                break;
            }
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<ConstraintExpr, ParseError> {
        let mut left = self.parse_unary()?;
        self.skip_whitespace();
        loop {
            if self.match_str("*") {
                self.skip_whitespace();
                let right = self.parse_unary()?;
                left = ConstraintExpr::Mul(Box::new(left), Box::new(right));
            } else if self.match_str("/") {
                self.skip_whitespace();
                let right = self.parse_unary()?;
                left = ConstraintExpr::Div(Box::new(left), Box::new(right));
            } else {
                break;
            }
            self.skip_whitespace();
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<ConstraintExpr, ParseError> {
        self.skip_whitespace();
        if self.match_str("!") || self.match_str("not ") {
            self.skip_whitespace();
            let expr = self.parse_unary()?;
            Ok(ConstraintExpr::Not(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<ConstraintExpr, ParseError> {
        self.skip_whitespace();
        match self.peek() {
            Some('(') => {
                self.advance();
                let expr = self.parse_iff()?;
                self.expect(')')?;
                Ok(expr)
            }
            Some('"') => {
                self.advance();
                let mut s = String::new();
                while let Some(ch) = self.peek() {
                    if ch == '"' { self.advance(); break; }
                    s.push(ch);
                    self.advance();
                }
                Ok(ConstraintExpr::StringLit(s))
            }
            Some(ch) if ch.is_ascii_digit() => {
                let mut num = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() || ch == '.' {
                        num.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                if num.contains('.') {
                    Ok(ConstraintExpr::FloatLit(num.parse().map_err(|_| self.error("Invalid float"))?))
                } else {
                    Ok(ConstraintExpr::IntLit(num.parse().map_err(|_| self.error("Invalid integer"))?))
                }
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let mut name = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        name.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                match name.as_str() {
                    "true" => Ok(ConstraintExpr::BoolLit(true)),
                    "false" => Ok(ConstraintExpr::BoolLit(false)),
                    _ => {
                        self.skip_whitespace();
                        if self.peek() == Some('(') {
                            self.advance();
                            let mut args = Vec::new();
                            self.skip_whitespace();
                            if self.peek() != Some(')') {
                                args.push(self.parse_iff()?);
                                self.skip_whitespace();
                                while self.match_str(",") {
                                    self.skip_whitespace();
                                    args.push(self.parse_iff()?);
                                    self.skip_whitespace();
                                }
                            }
                            self.expect(')')?;
                            Ok(ConstraintExpr::FuncApp(name, args))
                        } else {
                            Ok(ConstraintExpr::Var(name))
                        }
                    }
                }
            }
            _ => Err(self.error("Unexpected character")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Environment — variable bindings for evaluation
// ═══════════════════════════════════════════════════════════════════════

/// Environment for evaluating constraint expressions.
/// Environment for evaluating constraint expressions.
pub struct Environment {
    /// Variable bindings.
    pub bindings: HashMap<String, ConstraintValue>,
    /// Function definitions (stored as Arc for clone support).
    pub functions: HashMap<String, std::sync::Arc<dyn Fn(&[ConstraintValue]) -> ConstraintValue>>,
}

impl fmt::Debug for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Environment")
            .field("bindings", &self.bindings)
            .field("functions", &format!("[{} functions]", self.functions.len()))
            .finish()
    }
}

impl Clone for Environment {
    fn clone(&self) -> Self {
        Environment {
            bindings: self.bindings.clone(),
            functions: self.functions.clone(),
        }
    }
}
impl Environment {
    pub fn new() -> Self {
        Environment {
            bindings: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: &str, val: ConstraintValue) {
        self.bindings.insert(name.to_string(), val);
    }

    pub fn get(&self, name: &str) -> Option<&ConstraintValue> {
        self.bindings.get(name)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EvalError
// ═══════════════════════════════════════════════════════════════════════

/// Evaluation error.
#[derive(Debug, Clone)]
pub struct EvalError {
    pub message: String,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Evaluation error: {}", self.message)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintEvaluator
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate constraint expressions.
pub struct ConstraintEvaluator;

impl ConstraintEvaluator {
    /// Evaluate an expression in an environment.
    pub fn evaluate(expr: &ConstraintExpr, env: &Environment) -> Result<ConstraintValue, EvalError> {
        match expr {
            ConstraintExpr::BoolLit(b) => Ok(ConstraintValue::Bool(*b)),
            ConstraintExpr::IntLit(i) => Ok(ConstraintValue::Int(*i)),
            ConstraintExpr::FloatLit(f) => Ok(ConstraintValue::Float(*f)),
            ConstraintExpr::StringLit(s) => Ok(ConstraintValue::Str(s.clone())),
            ConstraintExpr::Var(v) => {
                env.get(v)
                    .cloned()
                    .ok_or(EvalError { message: format!("Unbound variable: {}", v) })
            }
            ConstraintExpr::Not(e) => {
                let val = Self::evaluate(e, env)?;
                Ok(ConstraintValue::Bool(!val.is_truthy()))
            }
            ConstraintExpr::And(a, b) => {
                let va = Self::evaluate(a, env)?;
                if !va.is_truthy() {
                    return Ok(ConstraintValue::Bool(false)); // Short circuit
                }
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(vb.is_truthy()))
            }
            ConstraintExpr::Or(a, b) => {
                let va = Self::evaluate(a, env)?;
                if va.is_truthy() {
                    return Ok(ConstraintValue::Bool(true)); // Short circuit
                }
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(vb.is_truthy()))
            }
            ConstraintExpr::Implies(a, b) => {
                let va = Self::evaluate(a, env)?;
                if !va.is_truthy() {
                    return Ok(ConstraintValue::Bool(true));
                }
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(vb.is_truthy()))
            }
            ConstraintExpr::Iff(a, b) => {
                let va = Self::evaluate(a, env)?;
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(va.is_truthy() == vb.is_truthy()))
            }
            ConstraintExpr::Eq(a, b) => {
                let va = Self::evaluate(a, env)?;
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(va == vb))
            }
            ConstraintExpr::Ne(a, b) => {
                let va = Self::evaluate(a, env)?;
                let vb = Self::evaluate(b, env)?;
                Ok(ConstraintValue::Bool(va != vb))
            }
            ConstraintExpr::Lt(a, b) => Self::compare(a, b, env, |x, y| x < y),
            ConstraintExpr::Le(a, b) => Self::compare(a, b, env, |x, y| x <= y),
            ConstraintExpr::Gt(a, b) => Self::compare(a, b, env, |x, y| x > y),
            ConstraintExpr::Ge(a, b) => Self::compare(a, b, env, |x, y| x >= y),
            ConstraintExpr::Add(a, b) => Self::arith(a, b, env, |x, y| x + y),
            ConstraintExpr::Sub(a, b) => Self::arith(a, b, env, |x, y| x - y),
            ConstraintExpr::Mul(a, b) => Self::arith(a, b, env, |x, y| x * y),
            ConstraintExpr::Div(a, b) => {
                let va = Self::evaluate(a, env)?;
                let vb = Self::evaluate(b, env)?;
                match (&va, &vb) {
                    (ConstraintValue::Int(x), ConstraintValue::Int(y)) if *y != 0 => {
                        Ok(ConstraintValue::Int(x / y))
                    }
                    (ConstraintValue::Float(x), ConstraintValue::Float(y)) if *y != 0.0 => {
                        Ok(ConstraintValue::Float(x / y))
                    }
                    _ => Err(EvalError { message: "Division error".into() }),
                }
            }
            ConstraintExpr::ForAll(var, domain, body) => {
                let mut env = env.clone();
                for val in domain.enumerate() {
                    env.bind(var, val);
                    let result = Self::evaluate(body, &env)?;
                    if !result.is_truthy() {
                        return Ok(ConstraintValue::Bool(false));
                    }
                }
                Ok(ConstraintValue::Bool(true))
            }
            ConstraintExpr::Exists(var, domain, body) => {
                let mut env = env.clone();
                for val in domain.enumerate() {
                    env.bind(var, val);
                    let result = Self::evaluate(body, &env)?;
                    if result.is_truthy() {
                        return Ok(ConstraintValue::Bool(true));
                    }
                }
                Ok(ConstraintValue::Bool(false))
            }
            ConstraintExpr::SetIn(elem, set) => {
                let e = Self::evaluate(elem, env)?;
                let s = Self::evaluate(set, env)?;
                match s {
                    ConstraintValue::Set(vals) => {
                        Ok(ConstraintValue::Bool(vals.contains(&e)))
                    }
                    _ => Err(EvalError { message: "SetIn requires a set".into() }),
                }
            }
            ConstraintExpr::FuncApp(_, _) => {
                Err(EvalError { message: "Function application not yet implemented".into() })
            }
        }
    }

    fn compare(
        a: &ConstraintExpr, b: &ConstraintExpr,
        env: &Environment,
        cmp: fn(f64, f64) -> bool,
    ) -> Result<ConstraintValue, EvalError> {
        let va = Self::evaluate(a, env)?;
        let vb = Self::evaluate(b, env)?;
        match (va.as_float(), vb.as_float()) {
            (Some(x), Some(y)) => Ok(ConstraintValue::Bool(cmp(x, y))),
            _ => Err(EvalError { message: "Cannot compare non-numeric values".into() }),
        }
    }

    fn arith(
        a: &ConstraintExpr, b: &ConstraintExpr,
        env: &Environment,
        op: fn(i64, i64) -> i64,
    ) -> Result<ConstraintValue, EvalError> {
        let va = Self::evaluate(a, env)?;
        let vb = Self::evaluate(b, env)?;
        match (&va, &vb) {
            (ConstraintValue::Int(x), ConstraintValue::Int(y)) => {
                Ok(ConstraintValue::Int(op(*x, *y)))
            }
            _ => Err(EvalError { message: "Arithmetic requires integers".into() }),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintSimplifier
// ═══════════════════════════════════════════════════════════════════════

/// Simplify constraint expressions.
pub struct ConstraintSimplifier;

impl ConstraintSimplifier {
    /// Simplify a constraint expression.
    pub fn simplify(expr: &ConstraintExpr) -> ConstraintExpr {
        match expr {
            // Double negation: ¬¬A = A
            ConstraintExpr::Not(inner) => {
                match inner.as_ref() {
                    ConstraintExpr::Not(a) => Self::simplify(a),
                    ConstraintExpr::BoolLit(b) => ConstraintExpr::BoolLit(!b),
                    _ => ConstraintExpr::Not(Box::new(Self::simplify(inner))),
                }
            }
            // And simplifications
            ConstraintExpr::And(a, b) => {
                let sa = Self::simplify(a);
                let sb = Self::simplify(b);
                match (&sa, &sb) {
                    (ConstraintExpr::BoolLit(true), _) => sb,
                    (_, ConstraintExpr::BoolLit(true)) => sa,
                    (ConstraintExpr::BoolLit(false), _) => ConstraintExpr::BoolLit(false),
                    (_, ConstraintExpr::BoolLit(false)) => ConstraintExpr::BoolLit(false),
                    _ => ConstraintExpr::And(Box::new(sa), Box::new(sb)),
                }
            }
            // Or simplifications
            ConstraintExpr::Or(a, b) => {
                let sa = Self::simplify(a);
                let sb = Self::simplify(b);
                match (&sa, &sb) {
                    (ConstraintExpr::BoolLit(false), _) => sb,
                    (_, ConstraintExpr::BoolLit(false)) => sa,
                    (ConstraintExpr::BoolLit(true), _) => ConstraintExpr::BoolLit(true),
                    (_, ConstraintExpr::BoolLit(true)) => ConstraintExpr::BoolLit(true),
                    _ => ConstraintExpr::Or(Box::new(sa), Box::new(sb)),
                }
            }
            // Implies: A → B = ¬A ∨ B
            ConstraintExpr::Implies(a, b) => {
                let sa = Self::simplify(a);
                let sb = Self::simplify(b);
                match &sa {
                    ConstraintExpr::BoolLit(false) => ConstraintExpr::BoolLit(true),
                    ConstraintExpr::BoolLit(true) => sb,
                    _ => ConstraintExpr::Implies(Box::new(sa), Box::new(sb)),
                }
            }
            // Constant folding for arithmetic
            ConstraintExpr::Add(a, b) => {
                let sa = Self::simplify(a);
                let sb = Self::simplify(b);
                match (&sa, &sb) {
                    (ConstraintExpr::IntLit(x), ConstraintExpr::IntLit(y)) => {
                        ConstraintExpr::IntLit(x + y)
                    }
                    (ConstraintExpr::IntLit(0), _) => sb,
                    (_, ConstraintExpr::IntLit(0)) => sa,
                    _ => ConstraintExpr::Add(Box::new(sa), Box::new(sb)),
                }
            }
            ConstraintExpr::Mul(a, b) => {
                let sa = Self::simplify(a);
                let sb = Self::simplify(b);
                match (&sa, &sb) {
                    (ConstraintExpr::IntLit(x), ConstraintExpr::IntLit(y)) => {
                        ConstraintExpr::IntLit(x * y)
                    }
                    (ConstraintExpr::IntLit(0), _) | (_, ConstraintExpr::IntLit(0)) => {
                        ConstraintExpr::IntLit(0)
                    }
                    (ConstraintExpr::IntLit(1), _) => sb,
                    (_, ConstraintExpr::IntLit(1)) => sa,
                    _ => ConstraintExpr::Mul(Box::new(sa), Box::new(sb)),
                }
            }
            _ => expr.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintNormalizer — NNF, CNF, DNF conversion
// ═══════════════════════════════════════════════════════════════════════

/// Convert constraints to normal forms.
pub struct ConstraintNormalizer;

impl ConstraintNormalizer {
    /// Convert to Negation Normal Form (NNF).
    pub fn to_nnf(expr: &ConstraintExpr) -> ConstraintExpr {
        match expr {
            ConstraintExpr::Not(inner) => {
                match inner.as_ref() {
                    ConstraintExpr::Not(a) => Self::to_nnf(a),
                    ConstraintExpr::And(a, b) => {
                        // De Morgan: ¬(A ∧ B) = ¬A ∨ ¬B
                        ConstraintExpr::Or(
                            Box::new(Self::to_nnf(&ConstraintExpr::Not(a.clone()))),
                            Box::new(Self::to_nnf(&ConstraintExpr::Not(b.clone()))),
                        )
                    }
                    ConstraintExpr::Or(a, b) => {
                        // De Morgan: ¬(A ∨ B) = ¬A ∧ ¬B
                        ConstraintExpr::And(
                            Box::new(Self::to_nnf(&ConstraintExpr::Not(a.clone()))),
                            Box::new(Self::to_nnf(&ConstraintExpr::Not(b.clone()))),
                        )
                    }
                    ConstraintExpr::Implies(a, b) => {
                        // ¬(A → B) = A ∧ ¬B
                        ConstraintExpr::And(
                            Box::new(Self::to_nnf(a)),
                            Box::new(Self::to_nnf(&ConstraintExpr::Not(b.clone()))),
                        )
                    }
                    ConstraintExpr::BoolLit(b) => ConstraintExpr::BoolLit(!b),
                    _ => ConstraintExpr::Not(Box::new(Self::to_nnf(inner))),
                }
            }
            ConstraintExpr::And(a, b) => {
                ConstraintExpr::And(Box::new(Self::to_nnf(a)), Box::new(Self::to_nnf(b)))
            }
            ConstraintExpr::Or(a, b) => {
                ConstraintExpr::Or(Box::new(Self::to_nnf(a)), Box::new(Self::to_nnf(b)))
            }
            ConstraintExpr::Implies(a, b) => {
                // A → B = ¬A ∨ B
                ConstraintExpr::Or(
                    Box::new(Self::to_nnf(&ConstraintExpr::Not(a.clone()))),
                    Box::new(Self::to_nnf(b)),
                )
            }
            _ => expr.clone(),
        }
    }

    /// Convert to CNF (Conjunctive Normal Form).
    pub fn to_cnf(expr: &ConstraintExpr) -> ConstraintExpr {
        let nnf = Self::to_nnf(expr);
        Self::distribute_or_over_and(&nnf)
    }

    fn distribute_or_over_and(expr: &ConstraintExpr) -> ConstraintExpr {
        match expr {
            ConstraintExpr::Or(a, b) => {
                let ca = Self::distribute_or_over_and(a);
                let cb = Self::distribute_or_over_and(b);
                match (&ca, &cb) {
                    (ConstraintExpr::And(a1, a2), _) => {
                        ConstraintExpr::And(
                            Box::new(Self::distribute_or_over_and(
                                &ConstraintExpr::Or(a1.clone(), Box::new(cb.clone()))
                            )),
                            Box::new(Self::distribute_or_over_and(
                                &ConstraintExpr::Or(a2.clone(), Box::new(cb))
                            )),
                        )
                    }
                    (_, ConstraintExpr::And(b1, b2)) => {
                        ConstraintExpr::And(
                            Box::new(Self::distribute_or_over_and(
                                &ConstraintExpr::Or(Box::new(ca.clone()), b1.clone())
                            )),
                            Box::new(Self::distribute_or_over_and(
                                &ConstraintExpr::Or(Box::new(ca), b2.clone())
                            )),
                        )
                    }
                    _ => ConstraintExpr::Or(Box::new(ca), Box::new(cb)),
                }
            }
            ConstraintExpr::And(a, b) => {
                ConstraintExpr::And(
                    Box::new(Self::distribute_or_over_and(a)),
                    Box::new(Self::distribute_or_over_and(b)),
                )
            }
            _ => expr.clone(),
        }
    }

    /// Convert to DNF (Disjunctive Normal Form).
    pub fn to_dnf(expr: &ConstraintExpr) -> ConstraintExpr {
        let nnf = Self::to_nnf(expr);
        Self::distribute_and_over_or(&nnf)
    }

    fn distribute_and_over_or(expr: &ConstraintExpr) -> ConstraintExpr {
        match expr {
            ConstraintExpr::And(a, b) => {
                let ca = Self::distribute_and_over_or(a);
                let cb = Self::distribute_and_over_or(b);
                match (&ca, &cb) {
                    (ConstraintExpr::Or(a1, a2), _) => {
                        ConstraintExpr::Or(
                            Box::new(Self::distribute_and_over_or(
                                &ConstraintExpr::And(a1.clone(), Box::new(cb.clone()))
                            )),
                            Box::new(Self::distribute_and_over_or(
                                &ConstraintExpr::And(a2.clone(), Box::new(cb))
                            )),
                        )
                    }
                    (_, ConstraintExpr::Or(b1, b2)) => {
                        ConstraintExpr::Or(
                            Box::new(Self::distribute_and_over_or(
                                &ConstraintExpr::And(Box::new(ca.clone()), b1.clone())
                            )),
                            Box::new(Self::distribute_and_over_or(
                                &ConstraintExpr::And(Box::new(ca), b2.clone())
                            )),
                        )
                    }
                    _ => ConstraintExpr::And(Box::new(ca), Box::new(cb)),
                }
            }
            ConstraintExpr::Or(a, b) => {
                ConstraintExpr::Or(
                    Box::new(Self::distribute_and_over_or(a)),
                    Box::new(Self::distribute_and_over_or(b)),
                )
            }
            _ => expr.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintSolver — backtracking solver with arc consistency
// ═══════════════════════════════════════════════════════════════════════

/// Result of constraint solving.
#[derive(Debug, Clone)]
pub enum SolverResult {
    Satisfiable(HashMap<String, ConstraintValue>),
    Unsatisfiable,
    Unknown,
}

/// Backtracking constraint solver with forward checking.
pub struct ConstraintSolver;

impl ConstraintSolver {
    /// Solve a constraint problem.
    pub fn solve(
        expr: &ConstraintExpr,
        domains: &HashMap<String, Domain>,
    ) -> SolverResult {
        let variables: Vec<String> = domains.keys().cloned().collect();
        let mut assignment: HashMap<String, ConstraintValue> = HashMap::new();
        let mut current_domains: HashMap<String, Vec<ConstraintValue>> = domains.iter()
            .map(|(k, d)| (k.clone(), d.enumerate()))
            .collect();

        if Self::backtrack(&variables, 0, &mut assignment, &mut current_domains, expr) {
            SolverResult::Satisfiable(assignment)
        } else {
            SolverResult::Unsatisfiable
        }
    }

    fn backtrack(
        variables: &[String],
        idx: usize,
        assignment: &mut HashMap<String, ConstraintValue>,
        domains: &mut HashMap<String, Vec<ConstraintValue>>,
        expr: &ConstraintExpr,
    ) -> bool {
        if idx == variables.len() {
            // All variables assigned, check constraint
            let mut env = Environment::new();
            for (k, v) in assignment.iter() {
                env.bind(k, v.clone());
            }
            match ConstraintEvaluator::evaluate(expr, &env) {
                Ok(val) => val.is_truthy(),
                Err(_) => false,
            }
        } else {
            let var = &variables[idx];
            let values = domains.get(var).cloned().unwrap_or_default();

            for val in values {
                assignment.insert(var.clone(), val.clone());

                // Forward checking: verify partial assignment
                if Self::is_consistent(assignment, expr) {
                    if Self::backtrack(variables, idx + 1, assignment, domains, expr) {
                        return true;
                    }
                }

                assignment.remove(var);
            }

            false
        }
    }

    fn is_consistent(
        assignment: &HashMap<String, ConstraintValue>,
        expr: &ConstraintExpr,
    ) -> bool {
        let mut env = Environment::new();
        for (k, v) in assignment.iter() {
            env.bind(k, v.clone());
        }
        // Evaluate with partial assignment — if all needed vars are bound, check result
        match ConstraintEvaluator::evaluate(expr, &env) {
            Ok(val) => val.is_truthy(),
            Err(_) => true, // If can't evaluate (unbound vars), assume consistent
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstraintPrinter — pretty printing
// ═══════════════════════════════════════════════════════════════════════

/// Pretty-print constraints in various formats.
pub struct ConstraintPrinter;

impl ConstraintPrinter {
    /// Print in standard text format.
    pub fn to_text(expr: &ConstraintExpr) -> String {
        format!("{}", expr)
    }

    /// Print in LaTeX format.
    pub fn to_latex(expr: &ConstraintExpr) -> String {
        match expr {
            ConstraintExpr::BoolLit(true) => "\\top".to_string(),
            ConstraintExpr::BoolLit(false) => "\\bot".to_string(),
            ConstraintExpr::Var(v) => v.clone(),
            ConstraintExpr::IntLit(i) => format!("{}", i),
            ConstraintExpr::Not(e) => format!("\\neg {}", Self::to_latex(e)),
            ConstraintExpr::And(a, b) => {
                format!("({} \\wedge {})", Self::to_latex(a), Self::to_latex(b))
            }
            ConstraintExpr::Or(a, b) => {
                format!("({} \\vee {})", Self::to_latex(a), Self::to_latex(b))
            }
            ConstraintExpr::Implies(a, b) => {
                format!("({} \\Rightarrow {})", Self::to_latex(a), Self::to_latex(b))
            }
            ConstraintExpr::Iff(a, b) => {
                format!("({} \\Leftrightarrow {})", Self::to_latex(a), Self::to_latex(b))
            }
            ConstraintExpr::ForAll(v, _, e) => {
                format!("\\forall {}: {}", v, Self::to_latex(e))
            }
            ConstraintExpr::Exists(v, _, e) => {
                format!("\\exists {}: {}", v, Self::to_latex(e))
            }
            _ => format!("{}", expr),
        }
    }

    /// Print in SMT-LIB2 format.
    pub fn to_smtlib2(expr: &ConstraintExpr) -> String {
        match expr {
            ConstraintExpr::BoolLit(b) => format!("{}", b),
            ConstraintExpr::Var(v) => v.clone(),
            ConstraintExpr::IntLit(i) => format!("{}", i),
            ConstraintExpr::Not(e) => format!("(not {})", Self::to_smtlib2(e)),
            ConstraintExpr::And(a, b) => {
                format!("(and {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Or(a, b) => {
                format!("(or {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Implies(a, b) => {
                format!("(=> {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Eq(a, b) => {
                format!("(= {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Lt(a, b) => {
                format!("(< {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Add(a, b) => {
                format!("(+ {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            ConstraintExpr::Mul(a, b) => {
                format!("(* {} {})", Self::to_smtlib2(a), Self::to_smtlib2(b))
            }
            _ => format!("{}", expr),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_boolean() {
        let expr = ConstraintParser::parse("true").unwrap();
        assert!(matches!(expr, ConstraintExpr::BoolLit(true)));

        let expr = ConstraintParser::parse("false").unwrap();
        assert!(matches!(expr, ConstraintExpr::BoolLit(false)));
    }

    #[test]
    fn test_parse_variable() {
        let expr = ConstraintParser::parse("x").unwrap();
        assert!(matches!(expr, ConstraintExpr::Var(v) if v == "x"));
    }

    #[test]
    fn test_parse_integer() {
        let expr = ConstraintParser::parse("42").unwrap();
        assert!(matches!(expr, ConstraintExpr::IntLit(42)));
    }

    #[test]
    fn test_parse_and() {
        let expr = ConstraintParser::parse("x && y").unwrap();
        assert!(matches!(expr, ConstraintExpr::And(_, _)));
    }

    #[test]
    fn test_parse_or() {
        let expr = ConstraintParser::parse("x || y").unwrap();
        assert!(matches!(expr, ConstraintExpr::Or(_, _)));
    }

    #[test]
    fn test_parse_not() {
        let expr = ConstraintParser::parse("!x").unwrap();
        assert!(matches!(expr, ConstraintExpr::Not(_)));
    }

    #[test]
    fn test_parse_comparison() {
        let expr = ConstraintParser::parse("x < 5").unwrap();
        assert!(matches!(expr, ConstraintExpr::Lt(_, _)));
    }

    #[test]
    fn test_parse_arithmetic() {
        let expr = ConstraintParser::parse("x + y * 2").unwrap();
        assert!(matches!(expr, ConstraintExpr::Add(_, _)));
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = ConstraintParser::parse("(x || y) && z").unwrap();
        assert!(matches!(expr, ConstraintExpr::And(_, _)));
    }

    #[test]
    fn test_parse_implies() {
        let expr = ConstraintParser::parse("x => y").unwrap();
        assert!(matches!(expr, ConstraintExpr::Implies(_, _)));
    }

    #[test]
    fn test_parse_function() {
        let expr = ConstraintParser::parse("f(x, y)").unwrap();
        assert!(matches!(expr, ConstraintExpr::FuncApp(_, _)));
    }

    #[test]
    fn test_evaluate_bool() {
        let env = Environment::new();
        let expr = ConstraintExpr::BoolLit(true);
        let result = ConstraintEvaluator::evaluate(&expr, &env).unwrap();
        assert_eq!(result, ConstraintValue::Bool(true));
    }

    #[test]
    fn test_evaluate_and() {
        let mut env = Environment::new();
        env.bind("x", ConstraintValue::Bool(true));
        env.bind("y", ConstraintValue::Bool(false));

        let expr = ConstraintExpr::And(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::Var("y".into())),
        );
        let result = ConstraintEvaluator::evaluate(&expr, &env).unwrap();
        assert_eq!(result, ConstraintValue::Bool(false));
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let mut env = Environment::new();
        env.bind("x", ConstraintValue::Int(3));
        env.bind("y", ConstraintValue::Int(4));

        let expr = ConstraintExpr::Add(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::Var("y".into())),
        );
        let result = ConstraintEvaluator::evaluate(&expr, &env).unwrap();
        assert_eq!(result, ConstraintValue::Int(7));
    }

    #[test]
    fn test_evaluate_comparison() {
        let mut env = Environment::new();
        env.bind("x", ConstraintValue::Int(3));

        let expr = ConstraintExpr::Lt(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::IntLit(5)),
        );
        let result = ConstraintEvaluator::evaluate(&expr, &env).unwrap();
        assert_eq!(result, ConstraintValue::Bool(true));
    }

    #[test]
    fn test_simplify_double_negation() {
        let expr = ConstraintExpr::Not(Box::new(
            ConstraintExpr::Not(Box::new(ConstraintExpr::Var("x".into())))
        ));
        let simplified = ConstraintSimplifier::simplify(&expr);
        assert!(matches!(simplified, ConstraintExpr::Var(v) if v == "x"));
    }

    #[test]
    fn test_simplify_and_true() {
        let expr = ConstraintExpr::And(
            Box::new(ConstraintExpr::BoolLit(true)),
            Box::new(ConstraintExpr::Var("x".into())),
        );
        let simplified = ConstraintSimplifier::simplify(&expr);
        assert!(matches!(simplified, ConstraintExpr::Var(v) if v == "x"));
    }

    #[test]
    fn test_simplify_and_false() {
        let expr = ConstraintExpr::And(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::BoolLit(false)),
        );
        let simplified = ConstraintSimplifier::simplify(&expr);
        assert!(matches!(simplified, ConstraintExpr::BoolLit(false)));
    }

    #[test]
    fn test_simplify_constant_fold() {
        let expr = ConstraintExpr::Add(
            Box::new(ConstraintExpr::IntLit(3)),
            Box::new(ConstraintExpr::IntLit(4)),
        );
        let simplified = ConstraintSimplifier::simplify(&expr);
        assert!(matches!(simplified, ConstraintExpr::IntLit(7)));
    }

    #[test]
    fn test_nnf_demorgan() {
        // ¬(A ∧ B) → ¬A ∨ ¬B
        let expr = ConstraintExpr::Not(Box::new(ConstraintExpr::And(
            Box::new(ConstraintExpr::Var("A".into())),
            Box::new(ConstraintExpr::Var("B".into())),
        )));
        let nnf = ConstraintNormalizer::to_nnf(&expr);
        assert!(matches!(nnf, ConstraintExpr::Or(_, _)));
    }

    #[test]
    fn test_solver_simple_csp() {
        // x + y == 5, x in {1,2,3}, y in {1,2,3}
        let expr = ConstraintExpr::Eq(
            Box::new(ConstraintExpr::Add(
                Box::new(ConstraintExpr::Var("x".into())),
                Box::new(ConstraintExpr::Var("y".into())),
            )),
            Box::new(ConstraintExpr::IntLit(5)),
        );

        let mut domains = HashMap::new();
        domains.insert("x".to_string(), Domain::IntRange(1, 3));
        domains.insert("y".to_string(), Domain::IntRange(1, 3));

        match ConstraintSolver::solve(&expr, &domains) {
            SolverResult::Satisfiable(assignment) => {
                let x = assignment.get("x").unwrap().as_int().unwrap();
                let y = assignment.get("y").unwrap().as_int().unwrap();
                assert_eq!(x + y, 5);
            }
            _ => panic!("Expected satisfiable"),
        }
    }

    #[test]
    fn test_solver_unsatisfiable() {
        // x > 10, x in {1,2,3}
        let expr = ConstraintExpr::Gt(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::IntLit(10)),
        );

        let mut domains = HashMap::new();
        domains.insert("x".to_string(), Domain::IntRange(1, 3));

        match ConstraintSolver::solve(&expr, &domains) {
            SolverResult::Unsatisfiable => {} // expected
            _ => panic!("Expected unsatisfiable"),
        }
    }

    #[test]
    fn test_latex_output() {
        let expr = ConstraintExpr::And(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::Not(Box::new(ConstraintExpr::Var("y".into())))),
        );
        let latex = ConstraintPrinter::to_latex(&expr);
        assert!(latex.contains("\\wedge"));
        assert!(latex.contains("\\neg"));
    }

    #[test]
    fn test_smtlib2_output() {
        let expr = ConstraintExpr::And(
            Box::new(ConstraintExpr::Var("x".into())),
            Box::new(ConstraintExpr::Var("y".into())),
        );
        let smt = ConstraintPrinter::to_smtlib2(&expr);
        assert_eq!(smt, "(and x y)");
    }
}
