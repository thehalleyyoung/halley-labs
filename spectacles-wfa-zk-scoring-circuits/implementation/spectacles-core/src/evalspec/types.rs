//! AST types for the EvalSpec DSL.
//!
//! This module defines the complete abstract syntax tree, type system,
//! metric specification structures, and validation logic for EvalSpec programs.

use std::fmt;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// 1. Source location types
// ---------------------------------------------------------------------------

/// Byte-level source span.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub file: String,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Default for Span {
    fn default() -> Self {
        Self {
            file: String::new(),
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }
}

impl Span {
    pub fn new(
        file: impl Into<String>,
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
    ) -> Self {
        Self {
            file: file.into(),
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }

    /// A synthetic span used for generated / built-in nodes.
    pub fn synthetic() -> Self {
        Self {
            file: "<synthetic>".into(),
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }

    /// Merge two spans into one that covers both.
    pub fn merge(&self, other: &Span) -> Span {
        let (start_line, start_col) = if self.start_line < other.start_line
            || (self.start_line == other.start_line && self.start_col <= other.start_col)
        {
            (self.start_line, self.start_col)
        } else {
            (other.start_line, other.start_col)
        };
        let (end_line, end_col) = if self.end_line > other.end_line
            || (self.end_line == other.end_line && self.end_col >= other.end_col)
        {
            (self.end_line, self.end_col)
        } else {
            (other.end_line, other.end_col)
        };
        Span {
            file: self.file.clone(),
            start_line,
            start_col,
            end_line,
            end_col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file, self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}

/// A wrapper that pairs any AST node `T` with its source [`Span`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    pub fn synthetic(node: T) -> Self {
        Self {
            node,
            span: Span::synthetic(),
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }

    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            node: &self.node,
            span: self.span.clone(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.node)
    }
}

// ---------------------------------------------------------------------------
// 2. Semiring type system
// ---------------------------------------------------------------------------

/// The supported semiring domains.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemiringType {
    /// Natural-number counting semiring (ℕ, +, ×, 0, 1).
    Counting,
    /// Boolean semiring ({0,1}, ∨, ∧, 0, 1).
    Boolean,
    /// Tropical semiring (ℝ ∪ {∞}, min, +, ∞, 0).
    Tropical,
    /// Counting semiring bounded above by `max_val`.
    BoundedCounting(u64),
    /// Real-number semiring (ℝ, +, ×, 0, 1).
    Real,
    /// Log-domain semiring (ℝ ∪ {-∞}, log-sum-exp, +, -∞, 0).
    LogDomain,
    /// Viterbi semiring ([0,1], max, ×, 0, 1).
    Viterbi,
    /// Goldilocks field (p = 2^64 − 2^32 + 1).
    Goldilocks,
}

impl SemiringType {
    /// Return the additive identity for this semiring.
    pub fn zero(&self) -> f64 {
        match self {
            SemiringType::Counting => 0.0,
            SemiringType::Boolean => 0.0,
            SemiringType::Tropical => f64::INFINITY,
            SemiringType::BoundedCounting(_) => 0.0,
            SemiringType::Real => 0.0,
            SemiringType::LogDomain => f64::NEG_INFINITY,
            SemiringType::Viterbi => 0.0,
            SemiringType::Goldilocks => 0.0,
        }
    }

    /// Return the multiplicative identity for this semiring.
    pub fn one(&self) -> f64 {
        match self {
            SemiringType::Counting => 1.0,
            SemiringType::Boolean => 1.0,
            SemiringType::Tropical => 0.0,
            SemiringType::BoundedCounting(_) => 1.0,
            SemiringType::Real => 1.0,
            SemiringType::LogDomain => 0.0,
            SemiringType::Viterbi => 1.0,
            SemiringType::Goldilocks => 1.0,
        }
    }

    /// Apply the semiring addition.
    pub fn add(&self, a: f64, b: f64) -> f64 {
        match self {
            SemiringType::Counting | SemiringType::Real => a + b,
            SemiringType::Boolean => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
            SemiringType::Tropical => a.min(b),
            SemiringType::BoundedCounting(max) => (a + b).min(*max as f64),
            SemiringType::LogDomain => {
                if a == f64::NEG_INFINITY {
                    b
                } else if b == f64::NEG_INFINITY {
                    a
                } else {
                    let mx = a.max(b);
                    mx + ((a - mx).exp() + (b - mx).exp()).ln()
                }
            }
            SemiringType::Viterbi => a.max(b),
            SemiringType::Goldilocks => {
                let p: u64 = 0xFFFF_FFFF_0000_0001u64;
                let sum = (a as u64).wrapping_add(b as u64) % p;
                sum as f64
            }
        }
    }

    /// Apply the semiring multiplication.
    pub fn mul(&self, a: f64, b: f64) -> f64 {
        match self {
            SemiringType::Counting | SemiringType::Real => a * b,
            SemiringType::Boolean => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
            SemiringType::Tropical => a + b,
            SemiringType::BoundedCounting(max) => (a * b).min(*max as f64),
            SemiringType::LogDomain => a + b,
            SemiringType::Viterbi => a * b,
            SemiringType::Goldilocks => {
                let p: u128 = (1u128 << 64) - (1u128 << 32) + 1;
                let prod = ((a as u64 as u128) * (b as u64 as u128)) % p;
                prod as f64
            }
        }
    }

    /// Whether this semiring is commutative (all built-in ones are).
    pub fn is_commutative(&self) -> bool {
        true
    }
}

impl fmt::Display for SemiringType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemiringType::Counting => write!(f, "Counting"),
            SemiringType::Boolean => write!(f, "Boolean"),
            SemiringType::Tropical => write!(f, "Tropical"),
            SemiringType::BoundedCounting(n) => write!(f, "BoundedCounting({n})"),
            SemiringType::Real => write!(f, "Real"),
            SemiringType::LogDomain => write!(f, "LogDomain"),
            SemiringType::Viterbi => write!(f, "Viterbi"),
            SemiringType::Goldilocks => write!(f, "Goldilocks"),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Base / Eval types
// ---------------------------------------------------------------------------

/// Primitive and compound data types for values in the DSL.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseType {
    String,
    Integer,
    Float,
    Bool,
    List(Box<BaseType>),
    Tuple(Vec<BaseType>),
    /// A single token (opaque id).
    Token,
    /// Ordered sequence of tokens.
    TokenSequence,
    /// An n-gram of the given order.
    NGram(usize),
}

impl BaseType {
    /// Returns `true` if the type is a scalar (non-compound).
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            BaseType::String | BaseType::Integer | BaseType::Float | BaseType::Bool
        )
    }

    /// Returns `true` for sequence-like types.
    pub fn is_sequence(&self) -> bool {
        matches!(
            self,
            BaseType::List(_) | BaseType::TokenSequence | BaseType::Tuple(_)
        )
    }

    /// Depth of nesting (lists/tuples).
    pub fn nesting_depth(&self) -> usize {
        match self {
            BaseType::List(inner) => 1 + inner.nesting_depth(),
            BaseType::Tuple(elems) => {
                1 + elems.iter().map(|e| e.nesting_depth()).max().unwrap_or(0)
            }
            _ => 0,
        }
    }
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaseType::String => write!(f, "String"),
            BaseType::Integer => write!(f, "Int"),
            BaseType::Float => write!(f, "Float"),
            BaseType::Bool => write!(f, "Bool"),
            BaseType::List(inner) => write!(f, "List<{inner}>"),
            BaseType::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            BaseType::Token => write!(f, "Token"),
            BaseType::TokenSequence => write!(f, "TokenSequence"),
            BaseType::NGram(n) => write!(f, "NGram({n})"),
        }
    }
}

/// Type that combines a base type with an optional semiring annotation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalType {
    /// Plain base type with no semiring annotation.
    Base(BaseType),
    /// A value that lives in the given semiring.
    Semiring(SemiringType),
    /// A function type.
    Function {
        params: Vec<EvalType>,
        ret: Box<EvalType>,
    },
    /// A metric type: takes (reference, candidate) and yields a semiring value.
    Metric {
        input: Box<EvalType>,
        output: Box<EvalType>,
    },
    /// A type variable (used during type inference).
    TypeVar(String),
    /// The unit type.
    Unit,
    /// An annotated base type carrying semiring context.
    Annotated {
        base: BaseType,
        semiring: SemiringType,
    },
}

impl EvalType {
    /// Create an EvalType from a BaseType (no semiring annotation).
    pub fn base(b: BaseType) -> Self {
        EvalType::Base(b)
    }

    /// Create an EvalType with base type and optional semiring.
    pub fn with_semiring(base: BaseType, semiring: Option<SemiringType>) -> Self {
        match semiring {
            Some(sr) => EvalType::Annotated { base, semiring: sr },
            None => EvalType::Base(base),
        }
    }

    /// Get the base type component.
    pub fn get_base(&self) -> BaseType {
        match self {
            EvalType::Base(b) => b.clone(),
            EvalType::Annotated { base, .. } => base.clone(),
            EvalType::Unit => BaseType::Tuple(vec![]),
            EvalType::Semiring(_) => BaseType::Float,
            EvalType::Function { .. } => BaseType::String,
            EvalType::Metric { .. } => BaseType::Float,
            EvalType::TypeVar(_) => BaseType::String,
        }
    }

    /// Get the semiring component, if applicable.
    pub fn get_semiring(&self) -> Option<SemiringType> {
        match self {
            EvalType::Semiring(s) => Some(s.clone()),
            EvalType::Annotated { semiring, .. } => Some(semiring.clone()),
            _ => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            EvalType::Base(BaseType::Integer)
                | EvalType::Base(BaseType::Float)
                | EvalType::Semiring(_)
        )
    }

    pub fn is_function(&self) -> bool {
        matches!(self, EvalType::Function { .. })
    }

    pub fn is_metric(&self) -> bool {
        matches!(self, EvalType::Metric { .. })
    }

    pub fn is_type_var(&self) -> bool {
        matches!(self, EvalType::TypeVar(_))
    }

    /// Substitute type variables according to a substitution map.
    pub fn apply_substitution(&self, subst: &Substitution) -> EvalType {
        match self {
            EvalType::TypeVar(name) => subst
                .mappings
                .get(name)
                .cloned()
                .unwrap_or_else(|| self.clone()),
            EvalType::Function { params, ret } => EvalType::Function {
                params: params.iter().map(|p| p.apply_substitution(subst)).collect(),
                ret: Box::new(ret.apply_substitution(subst)),
            },
            EvalType::Metric { input, output } => EvalType::Metric {
                input: Box::new(input.apply_substitution(subst)),
                output: Box::new(output.apply_substitution(subst)),
            },
            _ => self.clone(),
        }
    }

    /// Collect all free type-variable names.
    pub fn free_type_vars(&self) -> Vec<String> {
        match self {
            EvalType::TypeVar(n) => vec![n.clone()],
            EvalType::Function { params, ret } => {
                let mut vars: Vec<String> = params
                    .iter()
                    .flat_map(|p| p.free_type_vars())
                    .collect();
                vars.extend(ret.free_type_vars());
                vars.sort();
                vars.dedup();
                vars
            }
            EvalType::Metric { input, output } => {
                let mut vars = input.free_type_vars();
                vars.extend(output.free_type_vars());
                vars.sort();
                vars.dedup();
                vars
            }
            _ => vec![],
        }
    }
}

impl fmt::Display for EvalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalType::Base(b) => write!(f, "{b}"),
            EvalType::Semiring(s) => write!(f, "Semiring<{s}>"),
            EvalType::Function { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            EvalType::Metric { input, output } => {
                write!(f, "Metric<{input}, {output}>")
            }
            EvalType::TypeVar(v) => write!(f, "'{v}"),
            EvalType::Unit => write!(f, "()"),
            EvalType::Annotated { base, semiring } => {
                write!(f, "{base} @ {semiring}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Operators
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Min,
    Max,
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Concat,
    SemiringAdd,
    SemiringMul,
}

impl BinaryOp {
    /// Whether this operator produces a boolean result.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
        )
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod | BinaryOp::Pow
        )
    }

    pub fn is_logical(&self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// Precedence level (higher binds tighter).
    pub fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::Neq => 3,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Concat => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 6,
            BinaryOp::Pow | BinaryOp::Min | BinaryOp::Max => 7,
            BinaryOp::SemiringAdd => 5,
            BinaryOp::SemiringMul => 6,
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Pow => "^",
            BinaryOp::Min => "min",
            BinaryOp::Max => "max",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::Eq => "==",
            BinaryOp::Neq => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::Concat => "++",
            BinaryOp::SemiringAdd => "⊕",
            BinaryOp::SemiringMul => "⊗",
        };
        write!(f, "{s}")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
    /// Logical not.
    Not,
    /// Kleene star (for semiring closures).
    Star,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::Star => write!(f, "*"),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Literal values
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    String(String),
    Integer(i64),
    Float(OrderedFloat<f64>),
    Bool(bool),
}

impl Literal {
    pub fn type_of(&self) -> BaseType {
        match self {
            Literal::String(_) => BaseType::String,
            Literal::Integer(_) => BaseType::Integer,
            Literal::Float(_) => BaseType::Float,
            Literal::Bool(_) => BaseType::Bool,
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::String(s) => write!(f, "\"{s}\""),
            Literal::Integer(n) => write!(f, "{n}"),
            Literal::Float(v) => write!(f, "{v}"),
            Literal::Bool(b) => write!(f, "{b}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Expressions
// ---------------------------------------------------------------------------

/// Parameters for a lambda expression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LambdaParam {
    pub name: String,
    pub ty: Option<EvalType>,
    pub span: Span,
}

/// The central expression AST.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// A literal value.
    Literal(Literal),

    /// A variable reference.
    Variable(String),

    /// A binary operation.
    BinaryOp {
        op: BinaryOp,
        left: Box<Spanned<Expr>>,
        right: Box<Spanned<Expr>>,
    },

    /// A unary operation.
    UnaryOp {
        op: UnaryOp,
        operand: Box<Spanned<Expr>>,
    },

    /// A function call: `name(args...)`.
    FunctionCall {
        name: String,
        args: Vec<Spanned<Expr>>,
    },

    /// A method call: `receiver.method(args...)`.
    MethodCall {
        receiver: Box<Spanned<Expr>>,
        method: String,
        args: Vec<Spanned<Expr>>,
    },

    /// A lambda (anonymous function).
    Lambda {
        params: Vec<LambdaParam>,
        body: Box<Spanned<Expr>>,
    },

    /// A local let-binding: `let x = e1 in e2`.
    Let {
        name: String,
        ty: Option<EvalType>,
        value: Box<Spanned<Expr>>,
        body: Box<Spanned<Expr>>,
    },

    /// Conditional expression.
    If {
        condition: Box<Spanned<Expr>>,
        then_branch: Box<Spanned<Expr>>,
        else_branch: Box<Spanned<Expr>>,
    },

    /// Pattern-match expression.
    Match {
        scrutinee: Box<Spanned<Expr>>,
        arms: Vec<MatchArm>,
    },

    /// A block (sequence of expressions; value = last).
    Block(Vec<Spanned<Expr>>),

    /// Field access: `expr.field`.
    FieldAccess {
        expr: Box<Spanned<Expr>>,
        field: String,
    },

    /// Index access: `expr[index]`.
    IndexAccess {
        expr: Box<Spanned<Expr>>,
        index: Box<Spanned<Expr>>,
    },

    /// List literal: `[e1, e2, ...]`.
    ListLiteral(Vec<Spanned<Expr>>),

    /// Tuple literal: `(e1, e2, ...)`.
    TupleLiteral(Vec<Spanned<Expr>>),

    /// Aggregation over a collection with a semiring operation.
    Aggregate {
        op: AggregationOp,
        collection: Box<Spanned<Expr>>,
        /// Optional variable binding for the body.
        binding: Option<String>,
        body: Option<Box<Spanned<Expr>>>,
        semiring: Option<SemiringType>,
    },

    /// Extract n-grams from a token sequence.
    NGramExtract {
        input: Box<Spanned<Expr>>,
        n: usize,
    },

    /// Tokenize a string expression.
    TokenizeExpr {
        input: Box<Spanned<Expr>>,
        tokenizer: Option<String>,
    },

    /// Pattern matching on tokens / strings (regex or glob).
    MatchPattern {
        input: Box<Spanned<Expr>>,
        pattern: String,
        mode: MatchMode,
    },

    /// Cast between semiring types.
    SemiringCast {
        expr: Box<Spanned<Expr>>,
        from: SemiringType,
        to: SemiringType,
    },

    /// Clip a count to a bound (used in BLEU clipped counts).
    ClipCount {
        count: Box<Spanned<Expr>>,
        max_count: Box<Spanned<Expr>>,
    },

    /// Metric composition: apply metric_b to the output of metric_a.
    Compose {
        first: Box<Spanned<Expr>>,
        second: Box<Spanned<Expr>>,
    },
}

/// Mode for `MatchPattern` expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchMode {
    Regex,
    Glob,
    Exact,
    Contains,
}

impl fmt::Display for MatchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchMode::Regex => write!(f, "regex"),
            MatchMode::Glob => write!(f, "glob"),
            MatchMode::Exact => write!(f, "exact"),
            MatchMode::Contains => write!(f, "contains"),
        }
    }
}

impl Expr {
    /// Shorthand constructors.
    pub fn int(n: i64) -> Self {
        Expr::Literal(Literal::Integer(n))
    }

    pub fn float(v: f64) -> Self {
        Expr::Literal(Literal::Float(OrderedFloat(v)))
    }

    pub fn string(s: impl Into<String>) -> Self {
        Expr::Literal(Literal::String(s.into()))
    }

    pub fn bool_lit(b: bool) -> Self {
        Expr::Literal(Literal::Bool(b))
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expr::Variable(name.into())
    }

    pub fn binary(op: BinaryOp, left: Spanned<Expr>, right: Spanned<Expr>) -> Self {
        Expr::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn unary(op: UnaryOp, operand: Spanned<Expr>) -> Self {
        Expr::UnaryOp {
            op,
            operand: Box::new(operand),
        }
    }

    pub fn call(name: impl Into<String>, args: Vec<Spanned<Expr>>) -> Self {
        Expr::FunctionCall {
            name: name.into(),
            args,
        }
    }

    pub fn if_expr(
        condition: Spanned<Expr>,
        then_branch: Spanned<Expr>,
        else_branch: Spanned<Expr>,
    ) -> Self {
        Expr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }

    pub fn let_expr(
        name: impl Into<String>,
        ty: Option<EvalType>,
        value: Spanned<Expr>,
        body: Spanned<Expr>,
    ) -> Self {
        Expr::Let {
            name: name.into(),
            ty,
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Returns `true` when the expression is a "simple" leaf.
    pub fn is_leaf(&self) -> bool {
        matches!(self, Expr::Literal(_) | Expr::Variable(_))
    }

    /// Walk the immediate children.
    pub fn children(&self) -> Vec<&Spanned<Expr>> {
        match self {
            Expr::Literal(_) | Expr::Variable(_) => vec![],
            Expr::BinaryOp { left, right, .. } => vec![left.as_ref(), right.as_ref()],
            Expr::UnaryOp { operand, .. } => vec![operand.as_ref()],
            Expr::FunctionCall { args, .. } => args.iter().collect(),
            Expr::MethodCall {
                receiver, args, ..
            } => {
                let mut v = vec![receiver.as_ref()];
                v.extend(args.iter());
                v
            }
            Expr::Lambda { body, .. } => vec![body.as_ref()],
            Expr::Let { value, body, .. } => vec![value.as_ref(), body.as_ref()],
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => vec![
                condition.as_ref(),
                then_branch.as_ref(),
                else_branch.as_ref(),
            ],
            Expr::Match { scrutinee, arms } => {
                let mut v = vec![scrutinee.as_ref()];
                for arm in arms {
                    v.push(&arm.body);
                    if let Some(g) = &arm.guard {
                        v.push(g);
                    }
                }
                v
            }
            Expr::Block(exprs) => exprs.iter().collect(),
            Expr::FieldAccess { expr, .. } => vec![expr.as_ref()],
            Expr::IndexAccess { expr, index } => vec![expr.as_ref(), index.as_ref()],
            Expr::ListLiteral(elems) => elems.iter().collect(),
            Expr::TupleLiteral(elems) => elems.iter().collect(),
            Expr::Aggregate {
                collection, body, ..
            } => {
                let mut v = vec![collection.as_ref()];
                if let Some(b) = body {
                    v.push(b.as_ref());
                }
                v
            }
            Expr::NGramExtract { input, .. } => vec![input.as_ref()],
            Expr::TokenizeExpr { input, .. } => vec![input.as_ref()],
            Expr::MatchPattern { input, .. } => vec![input.as_ref()],
            Expr::SemiringCast { expr, .. } => vec![expr.as_ref()],
            Expr::ClipCount { count, max_count } => {
                vec![count.as_ref(), max_count.as_ref()]
            }
            Expr::Compose { first, second } => vec![first.as_ref(), second.as_ref()],
        }
    }

    /// Count all nodes in this expression tree.
    pub fn node_count(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(|c| c.node.node_count())
            .sum::<usize>()
    }

    /// Maximum depth of nesting.
    pub fn depth(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(|c| c.node.depth())
            .max()
            .unwrap_or(0)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(lit) => write!(f, "{lit}"),
            Expr::Variable(name) => write!(f, "{name}"),
            Expr::BinaryOp { op, left, right } => {
                write!(f, "({} {op} {})", left.node, right.node)
            }
            Expr::UnaryOp { op, operand } => write!(f, "({op}{})", operand.node),
            Expr::FunctionCall { name, args } => {
                write!(f, "{name}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a.node)?;
                }
                write!(f, ")")
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                write!(f, "{}.{method}(", receiver.node)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", a.node)?;
                }
                write!(f, ")")
            }
            Expr::Lambda { params, body } => {
                write!(f, "|")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.name)?;
                    if let Some(ty) = &p.ty {
                        write!(f, ": {ty}")?;
                    }
                }
                write!(f, "| {}", body.node)
            }
            Expr::Let {
                name, value, body, ..
            } => write!(f, "let {name} = {} in {}", value.node, body.node),
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => write!(
                f,
                "if {} then {} else {}",
                condition.node, then_branch.node, else_branch.node
            ),
            Expr::Match { scrutinee, arms } => {
                write!(f, "match {} {{ ", scrutinee.node)?;
                for arm in arms {
                    write!(f, "{} => {}, ", arm.pattern, arm.body.node)?;
                }
                write!(f, "}}")
            }
            Expr::Block(exprs) => {
                write!(f, "{{ ")?;
                for (i, e) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{}", e.node)?;
                }
                write!(f, " }}")
            }
            Expr::FieldAccess { expr, field } => write!(f, "{}.{field}", expr.node),
            Expr::IndexAccess { expr, index } => {
                write!(f, "{}[{}]", expr.node, index.node)
            }
            Expr::ListLiteral(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.node)?;
                }
                write!(f, "]")
            }
            Expr::TupleLiteral(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.node)?;
                }
                write!(f, ")")
            }
            Expr::Aggregate {
                op,
                collection,
                binding,
                body,
                ..
            } => {
                write!(f, "{op}(")?;
                if let Some(b) = binding {
                    write!(f, "{b} in ")?;
                }
                write!(f, "{}", collection.node)?;
                if let Some(bd) = body {
                    write!(f, " => {}", bd.node)?;
                }
                write!(f, ")")
            }
            Expr::NGramExtract { input, n } => {
                write!(f, "ngrams({}, {n})", input.node)
            }
            Expr::TokenizeExpr {
                input, tokenizer, ..
            } => {
                write!(f, "tokenize({})", input.node)?;
                if let Some(t) = tokenizer {
                    write!(f, "[{t}]")?;
                }
                Ok(())
            }
            Expr::MatchPattern {
                input,
                pattern,
                mode,
            } => write!(f, "match_pattern({}, \"{pattern}\", {mode})", input.node),
            Expr::SemiringCast { expr, from, to } => {
                write!(f, "cast<{from} -> {to}>({})", expr.node)
            }
            Expr::ClipCount { count, max_count } => {
                write!(f, "clip({}, {})", count.node, max_count.node)
            }
            Expr::Compose { first, second } => {
                write!(f, "({} >> {})", first.node, second.node)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Patterns (for match expressions)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    /// Matches anything, binds nothing.
    Wildcard,
    /// Matches anything, binds to name.
    Var(String),
    /// Matches a specific literal.
    Literal(Literal),
    /// Constructor pattern: `Some(inner)`, `None`, etc.
    Constructor {
        name: String,
        args: Vec<Spanned<Pattern>>,
    },
    /// Tuple pattern: `(a, b, c)`.
    Tuple(Vec<Spanned<Pattern>>),
    /// List pattern: `[head, ..tail]`.
    List {
        elems: Vec<Spanned<Pattern>>,
        rest: Option<Box<Spanned<Pattern>>>,
    },
    /// Guarded pattern: `p if cond`.
    Guard {
        pattern: Box<Spanned<Pattern>>,
        condition: Box<Spanned<Expr>>,
    },
}

impl Pattern {
    /// Collect all variable names bound by this pattern.
    pub fn bound_vars(&self) -> Vec<String> {
        match self {
            Pattern::Wildcard | Pattern::Literal(_) => vec![],
            Pattern::Var(name) => vec![name.clone()],
            Pattern::Constructor { args, .. } => {
                args.iter().flat_map(|a| a.node.bound_vars()).collect()
            }
            Pattern::Tuple(elems) => {
                elems.iter().flat_map(|e| e.node.bound_vars()).collect()
            }
            Pattern::List { elems, rest } => {
                let mut vars: Vec<String> =
                    elems.iter().flat_map(|e| e.node.bound_vars()).collect();
                if let Some(r) = rest {
                    vars.extend(r.node.bound_vars());
                }
                vars
            }
            Pattern::Guard { pattern, .. } => pattern.node.bound_vars(),
        }
    }

    /// Returns `true` if the pattern is irrefutable.
    pub fn is_irrefutable(&self) -> bool {
        match self {
            Pattern::Wildcard | Pattern::Var(_) => true,
            Pattern::Tuple(elems) => elems.iter().all(|e| e.node.is_irrefutable()),
            Pattern::Guard { .. } => false,
            Pattern::Literal(_) | Pattern::Constructor { .. } | Pattern::List { .. } => false,
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Var(name) => write!(f, "{name}"),
            Pattern::Literal(lit) => write!(f, "{lit}"),
            Pattern::Constructor { name, args } => {
                write!(f, "{name}")?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", a.node)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Pattern::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.node)?;
                }
                write!(f, ")")
            }
            Pattern::List { elems, rest } => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.node)?;
                }
                if let Some(r) = rest {
                    if !elems.is_empty() {
                        write!(f, ", ")?;
                    }
                    write!(f, "..{}", r.node)?;
                }
                write!(f, "]")
            }
            Pattern::Guard {
                pattern,
                condition,
            } => write!(f, "{} if {}", pattern.node, condition.node),
        }
    }
}

/// One arm in a `match` expression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Spanned<Pattern>,
    pub guard: Option<Spanned<Expr>>,
    pub body: Spanned<Expr>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// 8. Declarations & program structure
// ---------------------------------------------------------------------------

/// Attributes that can be attached to declarations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Doc(String),
    Deprecated(String),
    Test,
    Inline,
    SemiringHint(SemiringType),
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Doc(s) => write!(f, "#[doc = \"{s}\"]"),
            Attribute::Deprecated(s) => write!(f, "#[deprecated = \"{s}\"]"),
            Attribute::Test => write!(f, "#[test]"),
            Attribute::Inline => write!(f, "#[inline]"),
            Attribute::SemiringHint(s) => write!(f, "#[semiring_hint({s})]"),
        }
    }
}

/// Metric-level metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricMetadata {
    pub author: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub created_at: Option<DateTime<Utc>>,
}

impl MetricMetadata {
    pub fn empty() -> Self {
        Self {
            author: None,
            version: None,
            description: None,
            tags: vec![],
            created_at: None,
        }
    }
}

/// A metric declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricDecl {
    pub name: String,
    pub params: Vec<MetricParameter>,
    pub return_type: EvalType,
    pub body: Spanned<Expr>,
    pub attributes: Vec<Attribute>,
    pub metadata: MetricMetadata,
    pub span: Span,
}

/// A type alias declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeDecl {
    pub name: String,
    pub ty: EvalType,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}

/// A let binding at the top level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LetDecl {
    pub name: String,
    pub ty: Option<EvalType>,
    pub value: Spanned<Expr>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}

/// An import declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportDecl {
    pub path: Vec<String>,
    pub alias: Option<String>,
    pub items: ImportItems,
    pub span: Span,
}

/// What is being imported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImportItems {
    /// Import everything from the module.
    All,
    /// Import specific named items.
    Named(Vec<ImportItem>),
}

/// A single imported item, optionally renamed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportItem {
    pub name: String,
    pub alias: Option<String>,
}

/// Inline test declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestDecl {
    pub name: String,
    pub body: Spanned<Expr>,
    pub expected: TestExpectation,
    pub span: Span,
}

/// What a test expects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TestExpectation {
    /// Expect a specific value.
    Value(Literal),
    /// Expect the result to be approximately equal (for floats).
    Approx {
        value: OrderedFloat<f64>,
        tolerance: OrderedFloat<f64>,
    },
    /// Expect the expression to evaluate successfully (no crash).
    Success,
    /// Expect the expression to produce an error.
    Error(Option<String>),
}

/// Top-level declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Metric(MetricDecl),
    Type(TypeDecl),
    Let(LetDecl),
    Import(ImportDecl),
    Test(TestDecl),
}

impl Declaration {
    pub fn name(&self) -> Option<&str> {
        match self {
            Declaration::Metric(m) => Some(&m.name),
            Declaration::Type(t) => Some(&t.name),
            Declaration::Let(l) => Some(&l.name),
            Declaration::Import(_) => None,
            Declaration::Test(t) => Some(&t.name),
        }
    }

    pub fn span(&self) -> &Span {
        match self {
            Declaration::Metric(m) => &m.span,
            Declaration::Type(t) => &t.span,
            Declaration::Let(l) => &l.span,
            Declaration::Import(i) => &i.span,
            Declaration::Test(t) => &t.span,
        }
    }

    pub fn attributes(&self) -> &[Attribute] {
        match self {
            Declaration::Metric(m) => &m.attributes,
            Declaration::Type(t) => &t.attributes,
            Declaration::Let(l) => &l.attributes,
            Declaration::Import(_) => &[],
            Declaration::Test(_) => &[],
        }
    }

    pub fn is_metric(&self) -> bool {
        matches!(self, Declaration::Metric(_))
    }

    pub fn is_test(&self) -> bool {
        matches!(self, Declaration::Test(_))
    }
}

/// A full EvalSpec program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub declarations: Vec<Spanned<Declaration>>,
    pub span: Span,
}

impl Program {
    pub fn new(declarations: Vec<Spanned<Declaration>>, span: Span) -> Self {
        Self { declarations, span }
    }

    pub fn empty() -> Self {
        Self {
            declarations: vec![],
            span: Span::synthetic(),
        }
    }

    /// Iterate over metric declarations.
    pub fn metrics(&self) -> impl Iterator<Item = &MetricDecl> {
        self.declarations.iter().filter_map(|d| match &d.node {
            Declaration::Metric(m) => Some(m),
            _ => None,
        })
    }

    /// Iterate over type declarations.
    pub fn type_decls(&self) -> impl Iterator<Item = &TypeDecl> {
        self.declarations.iter().filter_map(|d| match &d.node {
            Declaration::Type(t) => Some(t),
            _ => None,
        })
    }

    /// Iterate over let declarations.
    pub fn let_decls(&self) -> impl Iterator<Item = &LetDecl> {
        self.declarations.iter().filter_map(|d| match &d.node {
            Declaration::Let(l) => Some(l),
            _ => None,
        })
    }

    /// Iterate over test declarations.
    pub fn tests(&self) -> impl Iterator<Item = &TestDecl> {
        self.declarations.iter().filter_map(|d| match &d.node {
            Declaration::Test(t) => Some(t),
            _ => None,
        })
    }

    /// Iterate over import declarations.
    pub fn imports(&self) -> impl Iterator<Item = &ImportDecl> {
        self.declarations.iter().filter_map(|d| match &d.node {
            Declaration::Import(i) => Some(i),
            _ => None,
        })
    }

    /// All declared names (metrics, types, lets).
    pub fn declared_names(&self) -> Vec<&str> {
        self.declarations
            .iter()
            .filter_map(|d| d.node.name())
            .collect()
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, decl) in self.declarations.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            match &decl.node {
                Declaration::Metric(m) => {
                    for attr in &m.attributes {
                        writeln!(f, "{attr}")?;
                    }
                    write!(f, "metric {}(", m.name)?;
                    for (j, p) in m.params.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", p.name, p.ty)?;
                    }
                    writeln!(f, ") -> {} = {}", m.return_type, m.body.node)?;
                }
                Declaration::Type(t) => {
                    writeln!(f, "type {} = {}", t.name, t.ty)?;
                }
                Declaration::Let(l) => {
                    write!(f, "let {}", l.name)?;
                    if let Some(ty) = &l.ty {
                        write!(f, ": {ty}")?;
                    }
                    writeln!(f, " = {}", l.value.node)?;
                }
                Declaration::Import(imp) => {
                    write!(f, "import {}", imp.path.join("::"))?;
                    match &imp.items {
                        ImportItems::All => write!(f, "::*")?,
                        ImportItems::Named(items) => {
                            write!(f, "::{{ ")?;
                            for (j, item) in items.iter().enumerate() {
                                if j > 0 {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{}", item.name)?;
                                if let Some(alias) = &item.alias {
                                    write!(f, " as {alias}")?;
                                }
                            }
                            write!(f, " }}")?;
                        }
                    }
                    if let Some(alias) = &imp.alias {
                        write!(f, " as {alias}")?;
                    }
                    writeln!(f)?;
                }
                Declaration::Test(t) => {
                    writeln!(f, "test \"{}\" = {}", t.name, t.body.node)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 9. Metric specification types
// ---------------------------------------------------------------------------

/// High-level metric type classification.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    ExactMatch,
    TokenF1,
    RegexMatch,
    BLEU,
    RougeN,
    RougeL,
    PassAtK,
    Custom,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::ExactMatch => write!(f, "ExactMatch"),
            MetricType::TokenF1 => write!(f, "TokenF1"),
            MetricType::RegexMatch => write!(f, "RegexMatch"),
            MetricType::BLEU => write!(f, "BLEU"),
            MetricType::RougeN => write!(f, "ROUGE-N"),
            MetricType::RougeL => write!(f, "ROUGE-L"),
            MetricType::PassAtK => write!(f, "Pass@K"),
            MetricType::Custom => write!(f, "Custom"),
        }
    }
}

/// A metric parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricParameter {
    pub name: String,
    pub ty: EvalType,
    pub default: Option<Spanned<Expr>>,
    pub span: Span,
}

/// A full metric specification (higher-level than MetricDecl).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricSpec {
    pub name: String,
    pub metric_type: MetricType,
    pub params: Vec<MetricParameter>,
    pub semiring: SemiringType,
    pub body: Spanned<Expr>,
}

/// BLEU smoothing methods.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SmoothingMethod {
    None,
    AddK(OrderedFloat<f64>),
    Floor(OrderedFloat<f64>),
    ChenCherry,
    Epsilon(OrderedFloat<f64>),
    NIST,
}

impl Default for SmoothingMethod {
    fn default() -> Self {
        SmoothingMethod::None
    }
}

impl fmt::Display for SmoothingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmoothingMethod::None => write!(f, "none"),
            SmoothingMethod::AddK(k) => write!(f, "add-k({k})"),
            SmoothingMethod::Floor(v) => write!(f, "floor({v})"),
            SmoothingMethod::ChenCherry => write!(f, "chen-cherry"),
            SmoothingMethod::Epsilon(e) => write!(f, "epsilon({e})"),
            SmoothingMethod::NIST => write!(f, "nist"),
        }
    }
}

/// Wrapper for n-gram order with validation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NGramOrder(usize);

impl NGramOrder {
    /// Create a new n-gram order. Returns `None` if `n == 0`.
    pub fn new(n: usize) -> Option<Self> {
        if n == 0 {
            None
        } else {
            Some(Self(n))
        }
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

impl fmt::Display for NGramOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Aggregation operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationOp {
    Sum,
    Product,
    Min,
    Max,
    Mean,
    HarmonicMean,
    GeometricMean,
    Count,
}

impl AggregationOp {
    /// The identity element for this aggregation when applied to f64.
    pub fn identity(&self) -> f64 {
        match self {
            AggregationOp::Sum | AggregationOp::Count => 0.0,
            AggregationOp::Product => 1.0,
            AggregationOp::Min => f64::INFINITY,
            AggregationOp::Max => f64::NEG_INFINITY,
            AggregationOp::Mean | AggregationOp::HarmonicMean | AggregationOp::GeometricMean => {
                0.0
            }
        }
    }

    /// Whether this operation is associative.
    pub fn is_associative(&self) -> bool {
        matches!(
            self,
            AggregationOp::Sum
                | AggregationOp::Product
                | AggregationOp::Min
                | AggregationOp::Max
                | AggregationOp::Count
        )
    }
}

impl fmt::Display for AggregationOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationOp::Sum => write!(f, "sum"),
            AggregationOp::Product => write!(f, "product"),
            AggregationOp::Min => write!(f, "min"),
            AggregationOp::Max => write!(f, "max"),
            AggregationOp::Mean => write!(f, "mean"),
            AggregationOp::HarmonicMean => write!(f, "harmonic_mean"),
            AggregationOp::GeometricMean => write!(f, "geometric_mean"),
            AggregationOp::Count => write!(f, "count"),
        }
    }
}

/// Configuration for BLEU scoring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BLEUConfig {
    pub max_n: usize,
    pub smoothing: SmoothingMethod,
    pub brevity_penalty: bool,
    pub weights: Vec<OrderedFloat<f64>>,
}

impl Default for BLEUConfig {
    fn default() -> Self {
        Self {
            max_n: 4,
            smoothing: SmoothingMethod::None,
            brevity_penalty: true,
            weights: vec![
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
                OrderedFloat(0.25),
            ],
        }
    }
}

impl BLEUConfig {
    /// Validate that weights sum to approximately 1.0 and max_n > 0.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_n == 0 {
            return Err("max_n must be > 0".into());
        }
        if self.weights.len() != self.max_n {
            return Err(format!(
                "weights length ({}) must equal max_n ({})",
                self.weights.len(),
                self.max_n
            ));
        }
        let sum: f64 = self.weights.iter().map(|w| w.0).sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("weights must sum to 1.0, got {sum}"));
        }
        Ok(())
    }
}

/// Scoring types for ROUGE.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RougeScoringType {
    Precision,
    Recall,
    FMeasure,
}

impl Default for RougeScoringType {
    fn default() -> Self {
        RougeScoringType::FMeasure
    }
}

impl fmt::Display for RougeScoringType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RougeScoringType::Precision => write!(f, "precision"),
            RougeScoringType::Recall => write!(f, "recall"),
            RougeScoringType::FMeasure => write!(f, "f-measure"),
        }
    }
}

/// Configuration for ROUGE scoring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RougeConfig {
    pub n_gram_size: usize,
    pub use_stemmer: bool,
    pub stopwords: Vec<String>,
    pub scoring_type: RougeScoringType,
}

impl Default for RougeConfig {
    fn default() -> Self {
        Self {
            n_gram_size: 1,
            use_stemmer: false,
            stopwords: vec![],
            scoring_type: RougeScoringType::FMeasure,
        }
    }
}

impl RougeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_gram_size == 0 {
            return Err("n_gram_size must be > 0".into());
        }
        Ok(())
    }
}

/// Configuration for pass@k evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PassAtKConfig {
    pub k_values: Vec<usize>,
    pub num_samples: usize,
}

impl Default for PassAtKConfig {
    fn default() -> Self {
        Self {
            k_values: vec![1, 10, 100],
            num_samples: 200,
        }
    }
}

impl PassAtKConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.k_values.is_empty() {
            return Err("k_values must not be empty".into());
        }
        for &k in &self.k_values {
            if k == 0 {
                return Err("k values must be > 0".into());
            }
            if k > self.num_samples {
                return Err(format!(
                    "k={k} exceeds num_samples={}",
                    self.num_samples
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 10. Type environment and constraint types
// ---------------------------------------------------------------------------

/// A binding in the type environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeBinding {
    pub name: String,
    pub ty: EvalType,
    pub mutable: bool,
}

/// Type environment: maps variable names to their types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TypeEnv {
    scopes: Vec<IndexMap<String, TypeBinding>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![IndexMap::new()],
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(IndexMap::new());
    }

    /// Pop the innermost scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Bind a variable in the current (innermost) scope.
    pub fn bind(&mut self, name: impl Into<String>, ty: EvalType, mutable: bool) {
        let name = name.into();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(
                name.clone(),
                TypeBinding {
                    name,
                    ty,
                    mutable,
                },
            );
        }
    }

    /// Look up a variable, searching from innermost to outermost scope.
    pub fn lookup(&self, name: &str) -> Option<&TypeBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding);
            }
        }
        None
    }

    /// Check if a name is bound in any scope.
    pub fn contains(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// All bindings visible at the current point (inner scopes shadow outer).
    pub fn all_bindings(&self) -> IndexMap<String, TypeBinding> {
        let mut merged = IndexMap::new();
        for scope in &self.scopes {
            for (k, v) in scope {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    /// Number of active scopes.
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Names bound in the current (innermost) scope only.
    pub fn current_scope_names(&self) -> Vec<&str> {
        self.scopes
            .last()
            .map(|s| s.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default()
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Type inference constraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Two types must be equal.
    Equal(EvalType, EvalType, Span),
    /// One type must be a subtype of another.
    Subtype(EvalType, EvalType, Span),
    /// A type must support semiring operations.
    HasSemiring(EvalType, Span),
    /// A type must be numeric.
    IsNumeric(EvalType, Span),
    /// A type must be a sequence.
    IsSequence(EvalType, Span),
    /// A type must be callable.
    IsCallable {
        ty: EvalType,
        args: Vec<EvalType>,
        ret: EvalType,
        span: Span,
    },
}

impl TypeConstraint {
    pub fn span(&self) -> &Span {
        match self {
            TypeConstraint::Equal(_, _, s) => s,
            TypeConstraint::Subtype(_, _, s) => s,
            TypeConstraint::HasSemiring(_, s) => s,
            TypeConstraint::IsNumeric(_, s) => s,
            TypeConstraint::IsSequence(_, s) => s,
            TypeConstraint::IsCallable { span, .. } => span,
        }
    }
}

/// A substitution mapping type variables to concrete types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Substitution {
    pub mappings: IndexMap<String, EvalType>,
}

impl Substitution {
    pub fn empty() -> Self {
        Self {
            mappings: IndexMap::new(),
        }
    }

    pub fn singleton(name: impl Into<String>, ty: EvalType) -> Self {
        let mut mappings = IndexMap::new();
        mappings.insert(name.into(), ty);
        Self { mappings }
    }

    pub fn insert(&mut self, name: impl Into<String>, ty: EvalType) {
        self.mappings.insert(name.into(), ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&EvalType> {
        self.mappings.get(name)
    }

    /// Compose two substitutions: apply `other` first, then `self`.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = IndexMap::new();
        for (k, v) in &other.mappings {
            result.insert(k.clone(), v.apply_substitution(self));
        }
        for (k, v) in &self.mappings {
            result.entry(k.clone()).or_insert_with(|| v.clone());
        }
        Substitution { mappings: result }
    }

    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    pub fn len(&self) -> usize {
        self.mappings.len()
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// 11. Validation errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Error)]
pub enum ValidationError {
    #[error("duplicate name `{name}` at {span}; first defined at {first_span}")]
    DuplicateName {
        name: String,
        span: Span,
        first_span: Span,
    },

    #[error("undefined variable `{name}` at {span}")]
    UndefinedVariable { name: String, span: Span },

    #[error("type mismatch at {span}: expected {expected}, found {found}")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("invalid n-gram order {order} at {span}: must be > 0")]
    InvalidNGramOrder { order: usize, span: Span },

    #[error("invalid BLEU config at {span}: {reason}")]
    InvalidBLEUConfig { reason: String, span: Span },

    #[error("invalid ROUGE config at {span}: {reason}")]
    InvalidRougeConfig { reason: String, span: Span },

    #[error("invalid pass@k config at {span}: {reason}")]
    InvalidPassAtKConfig { reason: String, span: Span },

    #[error("missing return type for metric `{name}` at {span}")]
    MissingReturnType { name: String, span: Span },

    #[error("empty program: no declarations")]
    EmptyProgram,

    #[error("non-exhaustive match at {span}: missing patterns")]
    NonExhaustiveMatch { span: Span },

    #[error("unreachable pattern at {span}")]
    UnreachablePattern { span: Span },

    #[error("cyclic type alias `{name}` at {span}")]
    CyclicTypeAlias { name: String, span: Span },

    #[error("invalid semiring cast from {from} to {to} at {span}")]
    InvalidSemiringCast {
        from: SemiringType,
        to: SemiringType,
        span: Span,
    },

    #[error("unsupported operation `{op}` for semiring {semiring} at {span}")]
    UnsupportedSemiringOp {
        op: String,
        semiring: SemiringType,
        span: Span,
    },

    #[error("arity mismatch at {span}: expected {expected} arguments, found {found}")]
    ArityMismatch {
        expected: usize,
        found: usize,
        span: Span,
    },

    #[error("import not found: `{path}` at {span}")]
    ImportNotFound { path: String, span: Span },

    #[error("duplicate parameter `{name}` in metric `{metric}` at {span}")]
    DuplicateParameter {
        name: String,
        metric: String,
        span: Span,
    },

    #[error("invalid default value for parameter `{name}` at {span}")]
    InvalidDefault { name: String, span: Span },

    #[error("empty block at {span}")]
    EmptyBlock { span: Span },

    #[error("invalid aggregate: {reason} at {span}")]
    InvalidAggregate { reason: String, span: Span },
}

impl ValidationError {
    pub fn span(&self) -> Option<&Span> {
        match self {
            ValidationError::DuplicateName { span, .. }
            | ValidationError::UndefinedVariable { span, .. }
            | ValidationError::TypeMismatch { span, .. }
            | ValidationError::InvalidNGramOrder { span, .. }
            | ValidationError::InvalidBLEUConfig { span, .. }
            | ValidationError::InvalidRougeConfig { span, .. }
            | ValidationError::InvalidPassAtKConfig { span, .. }
            | ValidationError::MissingReturnType { span, .. }
            | ValidationError::NonExhaustiveMatch { span }
            | ValidationError::UnreachablePattern { span }
            | ValidationError::CyclicTypeAlias { span, .. }
            | ValidationError::InvalidSemiringCast { span, .. }
            | ValidationError::UnsupportedSemiringOp { span, .. }
            | ValidationError::ArityMismatch { span, .. }
            | ValidationError::ImportNotFound { span, .. }
            | ValidationError::DuplicateParameter { span, .. }
            | ValidationError::InvalidDefault { span, .. }
            | ValidationError::EmptyBlock { span }
            | ValidationError::InvalidAggregate { span, .. } => Some(span),
            ValidationError::EmptyProgram => None,
        }
    }
}

// Serialize for ValidationError (needed for test round-trips)
impl Serialize for ValidationError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ValidationError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let _s = String::deserialize(deserializer)?;
        // We cannot reconstruct all variants from a string, so we use a fallback.
        Ok(ValidationError::EmptyProgram)
    }
}

// ---------------------------------------------------------------------------
// 12. Validation logic
// ---------------------------------------------------------------------------

/// Validate a program for well-formedness errors.
pub fn validate_program(program: &Program) -> Result<(), Vec<ValidationError>> {
    let mut errors: Vec<ValidationError> = Vec::new();

    if program.declarations.is_empty() {
        errors.push(ValidationError::EmptyProgram);
        return Err(errors);
    }

    // Check for duplicate top-level names.
    let mut seen_names: IndexMap<String, Span> = IndexMap::new();
    for decl in &program.declarations {
        if let Some(name) = decl.node.name() {
            if let Some(first_span) = seen_names.get(name) {
                errors.push(ValidationError::DuplicateName {
                    name: name.to_string(),
                    span: decl.span.clone(),
                    first_span: first_span.clone(),
                });
            } else {
                seen_names.insert(name.to_string(), decl.span.clone());
            }
        }
    }

    // Validate each declaration.
    for decl in &program.declarations {
        validate_declaration(&decl.node, &mut errors);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_declaration(decl: &Declaration, errors: &mut Vec<ValidationError>) {
    match decl {
        Declaration::Metric(m) => validate_metric_decl(m, errors),
        Declaration::Type(t) => validate_type_decl(t, errors),
        Declaration::Let(l) => validate_let_decl(l, errors),
        Declaration::Import(i) => validate_import_decl(i, errors),
        Declaration::Test(t) => validate_test_decl(t, errors),
    }
}

fn validate_metric_decl(m: &MetricDecl, errors: &mut Vec<ValidationError>) {
    // Check for duplicate parameter names.
    let mut param_names: IndexMap<String, Span> = IndexMap::new();
    for p in &m.params {
        if let Some(_first) = param_names.get(&p.name) {
            errors.push(ValidationError::DuplicateParameter {
                name: p.name.clone(),
                metric: m.name.clone(),
                span: p.span.clone(),
            });
        } else {
            param_names.insert(p.name.clone(), p.span.clone());
        }
    }

    // Validate the body expression.
    validate_expr(&m.body, errors);
}

fn validate_type_decl(_t: &TypeDecl, _errors: &mut Vec<ValidationError>) {
    // Type aliases are validated during type checking.
}

fn validate_let_decl(l: &LetDecl, errors: &mut Vec<ValidationError>) {
    validate_expr(&l.value, errors);
}

fn validate_import_decl(i: &ImportDecl, errors: &mut Vec<ValidationError>) {
    if i.path.is_empty() {
        errors.push(ValidationError::ImportNotFound {
            path: "".into(),
            span: i.span.clone(),
        });
    }
}

fn validate_test_decl(t: &TestDecl, errors: &mut Vec<ValidationError>) {
    validate_expr(&t.body, errors);
}

fn validate_expr(expr: &Spanned<Expr>, errors: &mut Vec<ValidationError>) {
    match &expr.node {
        Expr::Block(exprs) => {
            if exprs.is_empty() {
                errors.push(ValidationError::EmptyBlock {
                    span: expr.span.clone(),
                });
            }
            for e in exprs {
                validate_expr(e, errors);
            }
        }
        Expr::NGramExtract { input, n } => {
            if *n == 0 {
                errors.push(ValidationError::InvalidNGramOrder {
                    order: *n,
                    span: expr.span.clone(),
                });
            }
            validate_expr(input, errors);
        }
        Expr::Match { scrutinee, arms } => {
            validate_expr(scrutinee, errors);
            if arms.is_empty() {
                errors.push(ValidationError::NonExhaustiveMatch {
                    span: expr.span.clone(),
                });
            }
            for arm in arms {
                validate_expr(&arm.body, errors);
                if let Some(g) = &arm.guard {
                    validate_expr(g, errors);
                }
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            validate_expr(left, errors);
            validate_expr(right, errors);
        }
        Expr::UnaryOp { operand, .. } => {
            validate_expr(operand, errors);
        }
        Expr::FunctionCall { args, .. } => {
            for a in args {
                validate_expr(a, errors);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            validate_expr(receiver, errors);
            for a in args {
                validate_expr(a, errors);
            }
        }
        Expr::Lambda { body, .. } => {
            validate_expr(body, errors);
        }
        Expr::Let { value, body, .. } => {
            validate_expr(value, errors);
            validate_expr(body, errors);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_expr(condition, errors);
            validate_expr(then_branch, errors);
            validate_expr(else_branch, errors);
        }
        Expr::FieldAccess { expr: inner, .. } => {
            validate_expr(inner, errors);
        }
        Expr::IndexAccess { expr: inner, index } => {
            validate_expr(inner, errors);
            validate_expr(index, errors);
        }
        Expr::ListLiteral(elems) | Expr::TupleLiteral(elems) => {
            for e in elems {
                validate_expr(e, errors);
            }
        }
        Expr::Aggregate {
            collection, body, ..
        } => {
            validate_expr(collection, errors);
            if let Some(b) = body {
                validate_expr(b, errors);
            }
        }
        Expr::TokenizeExpr { input, .. } => {
            validate_expr(input, errors);
        }
        Expr::MatchPattern { input, .. } => {
            validate_expr(input, errors);
        }
        Expr::SemiringCast { expr: inner, .. } => {
            validate_expr(inner, errors);
        }
        Expr::ClipCount { count, max_count } => {
            validate_expr(count, errors);
            validate_expr(max_count, errors);
        }
        Expr::Compose { first, second } => {
            validate_expr(first, errors);
            validate_expr(second, errors);
        }
        // Leaves
        Expr::Literal(_) | Expr::Variable(_) => {}
    }
}

// ---------------------------------------------------------------------------
// 13. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn span(line: usize, col: usize) -> Span {
        Span::new("test.eval", line, col, line, col + 1)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::synthetic(node)
    }

    // ---- Span tests ----

    #[test]
    fn test_span_display() {
        let s = Span::new("foo.eval", 1, 0, 1, 5);
        assert_eq!(s.to_string(), "foo.eval:1:0-1:5");
    }

    #[test]
    fn test_span_merge() {
        let a = Span::new("f.eval", 1, 0, 1, 10);
        let b = Span::new("f.eval", 3, 5, 3, 15);
        let merged = a.merge(&b);
        assert_eq!(merged.start_line, 1);
        assert_eq!(merged.start_col, 0);
        assert_eq!(merged.end_line, 3);
        assert_eq!(merged.end_col, 15);
    }

    #[test]
    fn test_span_synthetic() {
        let s = Span::synthetic();
        assert_eq!(s.file, "<synthetic>");
        assert_eq!(s.start_line, 0);
    }

    // ---- Spanned tests ----

    #[test]
    fn test_spanned_map() {
        let s = Spanned::new(42i32, span(1, 0));
        let doubled = s.map(|x| x * 2);
        assert_eq!(doubled.node, 84);
        assert_eq!(doubled.span.start_line, 1);
    }

    #[test]
    fn test_spanned_as_ref() {
        let s = Spanned::new(String::from("hello"), span(1, 0));
        let r = s.as_ref();
        assert_eq!(r.node, &String::from("hello"));
    }

    // ---- SemiringType tests ----

    #[test]
    fn test_semiring_zero_one() {
        assert_eq!(SemiringType::Counting.zero(), 0.0);
        assert_eq!(SemiringType::Counting.one(), 1.0);
        assert_eq!(SemiringType::Tropical.zero(), f64::INFINITY);
        assert_eq!(SemiringType::Tropical.one(), 0.0);
        assert_eq!(SemiringType::Boolean.zero(), 0.0);
        assert_eq!(SemiringType::Boolean.one(), 1.0);
        assert_eq!(SemiringType::Viterbi.zero(), 0.0);
        assert_eq!(SemiringType::Viterbi.one(), 1.0);
        assert_eq!(SemiringType::LogDomain.zero(), f64::NEG_INFINITY);
        assert_eq!(SemiringType::LogDomain.one(), 0.0);
    }

    #[test]
    fn test_semiring_add_counting() {
        let sr = SemiringType::Counting;
        assert_eq!(sr.add(3.0, 4.0), 7.0);
    }

    #[test]
    fn test_semiring_mul_counting() {
        let sr = SemiringType::Counting;
        assert_eq!(sr.mul(3.0, 4.0), 12.0);
    }

    #[test]
    fn test_semiring_add_boolean() {
        let sr = SemiringType::Boolean;
        assert_eq!(sr.add(0.0, 0.0), 0.0);
        assert_eq!(sr.add(1.0, 0.0), 1.0);
        assert_eq!(sr.add(0.0, 1.0), 1.0);
        assert_eq!(sr.add(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_semiring_mul_boolean() {
        let sr = SemiringType::Boolean;
        assert_eq!(sr.mul(0.0, 0.0), 0.0);
        assert_eq!(sr.mul(1.0, 0.0), 0.0);
        assert_eq!(sr.mul(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_semiring_tropical() {
        let sr = SemiringType::Tropical;
        assert_eq!(sr.add(3.0, 5.0), 3.0); // min
        assert_eq!(sr.mul(3.0, 5.0), 8.0); // +
    }

    #[test]
    fn test_semiring_viterbi() {
        let sr = SemiringType::Viterbi;
        assert_eq!(sr.add(0.3, 0.7), 0.7); // max
        let product = sr.mul(0.5, 0.4);
        assert!((product - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_semiring_bounded_counting() {
        let sr = SemiringType::BoundedCounting(10);
        assert_eq!(sr.add(7.0, 5.0), 10.0); // clamped
        assert_eq!(sr.add(3.0, 4.0), 7.0);
    }

    #[test]
    fn test_semiring_display() {
        assert_eq!(SemiringType::Real.to_string(), "Real");
        assert_eq!(
            SemiringType::BoundedCounting(42).to_string(),
            "BoundedCounting(42)"
        );
        assert_eq!(SemiringType::Goldilocks.to_string(), "Goldilocks");
    }

    #[test]
    fn test_semiring_is_commutative() {
        assert!(SemiringType::Counting.is_commutative());
        assert!(SemiringType::Tropical.is_commutative());
    }

    // ---- BaseType tests ----

    #[test]
    fn test_base_type_is_scalar() {
        assert!(BaseType::String.is_scalar());
        assert!(BaseType::Integer.is_scalar());
        assert!(BaseType::Float.is_scalar());
        assert!(BaseType::Bool.is_scalar());
        assert!(!BaseType::List(Box::new(BaseType::Integer)).is_scalar());
        assert!(!BaseType::Token.is_scalar());
    }

    #[test]
    fn test_base_type_is_sequence() {
        assert!(BaseType::List(Box::new(BaseType::Integer)).is_sequence());
        assert!(BaseType::TokenSequence.is_sequence());
        assert!(BaseType::Tuple(vec![BaseType::Integer]).is_sequence());
        assert!(!BaseType::Integer.is_sequence());
    }

    #[test]
    fn test_base_type_nesting_depth() {
        assert_eq!(BaseType::Integer.nesting_depth(), 0);
        assert_eq!(
            BaseType::List(Box::new(BaseType::Integer)).nesting_depth(),
            1
        );
        assert_eq!(
            BaseType::List(Box::new(BaseType::List(Box::new(BaseType::Float)))).nesting_depth(),
            2
        );
        assert_eq!(
            BaseType::Tuple(vec![
                BaseType::Integer,
                BaseType::List(Box::new(BaseType::Bool))
            ])
            .nesting_depth(),
            2
        );
    }

    #[test]
    fn test_base_type_display() {
        assert_eq!(BaseType::String.to_string(), "String");
        assert_eq!(BaseType::NGram(3).to_string(), "NGram(3)");
        assert_eq!(
            BaseType::List(Box::new(BaseType::Integer)).to_string(),
            "List<Int>"
        );
        assert_eq!(
            BaseType::Tuple(vec![BaseType::Integer, BaseType::Bool]).to_string(),
            "(Int, Bool)"
        );
    }

    // ---- EvalType tests ----

    #[test]
    fn test_eval_type_is_numeric() {
        assert!(EvalType::Base(BaseType::Integer).is_numeric());
        assert!(EvalType::Base(BaseType::Float).is_numeric());
        assert!(EvalType::Semiring(SemiringType::Real).is_numeric());
        assert!(!EvalType::Base(BaseType::String).is_numeric());
    }

    #[test]
    fn test_eval_type_is_function() {
        let fn_ty = EvalType::Function {
            params: vec![EvalType::Base(BaseType::Integer)],
            ret: Box::new(EvalType::Base(BaseType::Bool)),
        };
        assert!(fn_ty.is_function());
        assert!(!EvalType::Unit.is_function());
    }

    #[test]
    fn test_eval_type_free_vars() {
        let ty = EvalType::Function {
            params: vec![EvalType::TypeVar("a".into())],
            ret: Box::new(EvalType::TypeVar("b".into())),
        };
        let vars = ty.free_type_vars();
        assert_eq!(vars, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_eval_type_substitution() {
        let ty = EvalType::TypeVar("a".into());
        let mut subst = Substitution::empty();
        subst.insert("a", EvalType::Base(BaseType::Integer));
        let result = ty.apply_substitution(&subst);
        assert_eq!(result, EvalType::Base(BaseType::Integer));
    }

    #[test]
    fn test_eval_type_display() {
        assert_eq!(EvalType::Unit.to_string(), "()");
        assert_eq!(
            EvalType::Semiring(SemiringType::Boolean).to_string(),
            "Semiring<Boolean>"
        );
        assert_eq!(EvalType::TypeVar("x".into()).to_string(), "'x");
        assert_eq!(
            EvalType::Annotated {
                base: BaseType::Float,
                semiring: SemiringType::Real,
            }
            .to_string(),
            "Float @ Real"
        );
    }

    // ---- Operator tests ----

    #[test]
    fn test_binary_op_classification() {
        assert!(BinaryOp::Add.is_arithmetic());
        assert!(BinaryOp::Eq.is_comparison());
        assert!(BinaryOp::And.is_logical());
        assert!(!BinaryOp::Add.is_comparison());
        assert!(!BinaryOp::Eq.is_arithmetic());
    }

    #[test]
    fn test_binary_op_precedence() {
        assert!(BinaryOp::Mul.precedence() > BinaryOp::Add.precedence());
        assert!(BinaryOp::Add.precedence() > BinaryOp::Eq.precedence());
        assert!(BinaryOp::Eq.precedence() > BinaryOp::And.precedence());
        assert!(BinaryOp::And.precedence() > BinaryOp::Or.precedence());
    }

    #[test]
    fn test_binary_op_display() {
        assert_eq!(BinaryOp::Add.to_string(), "+");
        assert_eq!(BinaryOp::Neq.to_string(), "!=");
        assert_eq!(BinaryOp::Min.to_string(), "min");
    }

    #[test]
    fn test_unary_op_display() {
        assert_eq!(UnaryOp::Neg.to_string(), "-");
        assert_eq!(UnaryOp::Not.to_string(), "!");
        assert_eq!(UnaryOp::Star.to_string(), "*");
    }

    // ---- Literal tests ----

    #[test]
    fn test_literal_type_of() {
        assert_eq!(Literal::String("hi".into()).type_of(), BaseType::String);
        assert_eq!(Literal::Integer(42).type_of(), BaseType::Integer);
        assert_eq!(
            Literal::Float(OrderedFloat(3.14)).type_of(),
            BaseType::Float
        );
        assert_eq!(Literal::Bool(true).type_of(), BaseType::Bool);
    }

    #[test]
    fn test_literal_display() {
        assert_eq!(Literal::String("hello".into()).to_string(), "\"hello\"");
        assert_eq!(Literal::Integer(42).to_string(), "42");
        assert_eq!(Literal::Bool(false).to_string(), "false");
    }

    // ---- Expression tests ----

    #[test]
    fn test_expr_constructors() {
        assert_eq!(Expr::int(42), Expr::Literal(Literal::Integer(42)));
        assert_eq!(
            Expr::float(3.14),
            Expr::Literal(Literal::Float(OrderedFloat(3.14)))
        );
        assert_eq!(
            Expr::string("hi"),
            Expr::Literal(Literal::String("hi".into()))
        );
        assert_eq!(Expr::bool_lit(true), Expr::Literal(Literal::Bool(true)));
        assert_eq!(Expr::var("x"), Expr::Variable("x".into()));
    }

    #[test]
    fn test_expr_is_leaf() {
        assert!(Expr::int(1).is_leaf());
        assert!(Expr::var("x").is_leaf());
        assert!(!Expr::call("f", vec![]).is_leaf());
    }

    #[test]
    fn test_expr_children_literal() {
        assert!(Expr::int(1).children().is_empty());
    }

    #[test]
    fn test_expr_children_binary() {
        let e = Expr::binary(
            BinaryOp::Add,
            spanned(Expr::int(1)),
            spanned(Expr::int(2)),
        );
        assert_eq!(e.children().len(), 2);
    }

    #[test]
    fn test_expr_node_count() {
        let leaf = Expr::int(1);
        assert_eq!(leaf.node_count(), 1);

        let binop = Expr::binary(
            BinaryOp::Add,
            spanned(Expr::int(1)),
            spanned(Expr::int(2)),
        );
        assert_eq!(binop.node_count(), 3);
    }

    #[test]
    fn test_expr_depth() {
        assert_eq!(Expr::int(1).depth(), 1);

        let nested = Expr::binary(
            BinaryOp::Add,
            spanned(Expr::binary(
                BinaryOp::Mul,
                spanned(Expr::int(1)),
                spanned(Expr::int(2)),
            )),
            spanned(Expr::int(3)),
        );
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn test_expr_display_literal() {
        assert_eq!(Expr::int(42).to_string(), "42");
    }

    #[test]
    fn test_expr_display_binary() {
        let e = Expr::binary(
            BinaryOp::Add,
            spanned(Expr::var("x")),
            spanned(Expr::int(1)),
        );
        assert_eq!(e.to_string(), "(x + 1)");
    }

    #[test]
    fn test_expr_display_function_call() {
        let e = Expr::call("f", vec![spanned(Expr::int(1)), spanned(Expr::int(2))]);
        assert_eq!(e.to_string(), "f(1, 2)");
    }

    #[test]
    fn test_expr_display_if() {
        let e = Expr::if_expr(
            spanned(Expr::bool_lit(true)),
            spanned(Expr::int(1)),
            spanned(Expr::int(0)),
        );
        assert_eq!(e.to_string(), "if true then 1 else 0");
    }

    #[test]
    fn test_expr_display_let() {
        let e = Expr::let_expr("x", None, spanned(Expr::int(5)), spanned(Expr::var("x")));
        assert_eq!(e.to_string(), "let x = 5 in x");
    }

    #[test]
    fn test_expr_display_list_literal() {
        let e = Expr::ListLiteral(vec![spanned(Expr::int(1)), spanned(Expr::int(2))]);
        assert_eq!(e.to_string(), "[1, 2]");
    }

    #[test]
    fn test_expr_display_lambda() {
        let e = Expr::Lambda {
            params: vec![LambdaParam {
                name: "x".into(),
                ty: None,
                span: Span::synthetic(),
            }],
            body: Box::new(spanned(Expr::var("x"))),
        };
        assert_eq!(e.to_string(), "|x| x");
    }

    #[test]
    fn test_expr_aggregate() {
        let e = Expr::Aggregate {
            op: AggregationOp::Sum,
            collection: Box::new(spanned(Expr::var("xs"))),
            binding: Some("x".into()),
            body: Some(Box::new(spanned(Expr::var("x")))),
            semiring: None,
        };
        assert_eq!(e.to_string(), "sum(x in xs => x)");
    }

    #[test]
    fn test_expr_ngram_extract() {
        let e = Expr::NGramExtract {
            input: Box::new(spanned(Expr::var("tokens"))),
            n: 3,
        };
        assert_eq!(e.to_string(), "ngrams(tokens, 3)");
    }

    #[test]
    fn test_expr_compose() {
        let e = Expr::Compose {
            first: Box::new(spanned(Expr::var("tokenize"))),
            second: Box::new(spanned(Expr::var("exact_match"))),
        };
        assert_eq!(e.to_string(), "(tokenize >> exact_match)");
    }

    #[test]
    fn test_expr_semiring_cast() {
        let e = Expr::SemiringCast {
            expr: Box::new(spanned(Expr::int(1))),
            from: SemiringType::Counting,
            to: SemiringType::Boolean,
        };
        assert_eq!(e.to_string(), "cast<Counting -> Boolean>(1)");
    }

    #[test]
    fn test_expr_clip_count() {
        let e = Expr::ClipCount {
            count: Box::new(spanned(Expr::int(5))),
            max_count: Box::new(spanned(Expr::int(3))),
        };
        assert_eq!(e.to_string(), "clip(5, 3)");
    }

    #[test]
    fn test_expr_method_call() {
        let e = Expr::MethodCall {
            receiver: Box::new(spanned(Expr::var("list"))),
            method: "map".into(),
            args: vec![spanned(Expr::var("f"))],
        };
        assert_eq!(e.to_string(), "list.map(f)");
    }

    #[test]
    fn test_expr_field_access() {
        let e = Expr::FieldAccess {
            expr: Box::new(spanned(Expr::var("config"))),
            field: "max_n".into(),
        };
        assert_eq!(e.to_string(), "config.max_n");
    }

    #[test]
    fn test_expr_index_access() {
        let e = Expr::IndexAccess {
            expr: Box::new(spanned(Expr::var("arr"))),
            index: Box::new(spanned(Expr::int(0))),
        };
        assert_eq!(e.to_string(), "arr[0]");
    }

    #[test]
    fn test_expr_match_pattern() {
        let e = Expr::MatchPattern {
            input: Box::new(spanned(Expr::var("s"))),
            pattern: "^hello".into(),
            mode: MatchMode::Regex,
        };
        assert_eq!(e.to_string(), "match_pattern(s, \"^hello\", regex)");
    }

    #[test]
    fn test_expr_tokenize() {
        let e = Expr::TokenizeExpr {
            input: Box::new(spanned(Expr::string("hello world"))),
            tokenizer: Some("whitespace".into()),
        };
        assert_eq!(e.to_string(), "tokenize(\"hello world\")[whitespace]");
    }

    #[test]
    fn test_expr_block() {
        let e = Expr::Block(vec![spanned(Expr::int(1)), spanned(Expr::int(2))]);
        assert_eq!(e.to_string(), "{ 1; 2 }");
    }

    #[test]
    fn test_expr_tuple_literal() {
        let e = Expr::TupleLiteral(vec![spanned(Expr::int(1)), spanned(Expr::bool_lit(true))]);
        assert_eq!(e.to_string(), "(1, true)");
    }

    #[test]
    fn test_expr_unary() {
        let e = Expr::unary(UnaryOp::Neg, spanned(Expr::int(5)));
        assert_eq!(e.to_string(), "(-5)");
    }

    // ---- Pattern tests ----

    #[test]
    fn test_pattern_wildcard() {
        let p = Pattern::Wildcard;
        assert!(p.bound_vars().is_empty());
        assert!(p.is_irrefutable());
        assert_eq!(p.to_string(), "_");
    }

    #[test]
    fn test_pattern_var() {
        let p = Pattern::Var("x".into());
        assert_eq!(p.bound_vars(), vec!["x"]);
        assert!(p.is_irrefutable());
        assert_eq!(p.to_string(), "x");
    }

    #[test]
    fn test_pattern_literal() {
        let p = Pattern::Literal(Literal::Integer(42));
        assert!(p.bound_vars().is_empty());
        assert!(!p.is_irrefutable());
        assert_eq!(p.to_string(), "42");
    }

    #[test]
    fn test_pattern_constructor() {
        let p = Pattern::Constructor {
            name: "Some".into(),
            args: vec![spanned(Pattern::Var("x".into()))],
        };
        assert_eq!(p.bound_vars(), vec!["x"]);
        assert!(!p.is_irrefutable());
        assert_eq!(p.to_string(), "Some(x)");
    }

    #[test]
    fn test_pattern_tuple() {
        let p = Pattern::Tuple(vec![
            spanned(Pattern::Var("a".into())),
            spanned(Pattern::Var("b".into())),
        ]);
        assert_eq!(p.bound_vars(), vec!["a", "b"]);
        assert!(p.is_irrefutable());
        assert_eq!(p.to_string(), "(a, b)");
    }

    #[test]
    fn test_pattern_list() {
        let p = Pattern::List {
            elems: vec![spanned(Pattern::Var("h".into()))],
            rest: Some(Box::new(spanned(Pattern::Var("t".into())))),
        };
        assert_eq!(p.bound_vars(), vec!["h", "t"]);
        assert!(!p.is_irrefutable());
        assert_eq!(p.to_string(), "[h, ..t]");
    }

    #[test]
    fn test_pattern_guard() {
        let p = Pattern::Guard {
            pattern: Box::new(spanned(Pattern::Var("x".into()))),
            condition: Box::new(spanned(Expr::binary(
                BinaryOp::Gt,
                spanned(Expr::var("x")),
                spanned(Expr::int(0)),
            ))),
        };
        assert_eq!(p.bound_vars(), vec!["x"]);
        assert!(!p.is_irrefutable());
    }

    // ---- Attribute tests ----

    #[test]
    fn test_attribute_display() {
        assert_eq!(
            Attribute::Doc("A metric".into()).to_string(),
            "#[doc = \"A metric\"]"
        );
        assert_eq!(Attribute::Test.to_string(), "#[test]");
        assert_eq!(Attribute::Inline.to_string(), "#[inline]");
        assert_eq!(
            Attribute::Deprecated("use v2".into()).to_string(),
            "#[deprecated = \"use v2\"]"
        );
        assert_eq!(
            Attribute::SemiringHint(SemiringType::Boolean).to_string(),
            "#[semiring_hint(Boolean)]"
        );
    }

    // ---- MetricMetadata tests ----

    #[test]
    fn test_metric_metadata_empty() {
        let m = MetricMetadata::empty();
        assert!(m.author.is_none());
        assert!(m.version.is_none());
        assert!(m.description.is_none());
        assert!(m.tags.is_empty());
        assert!(m.created_at.is_none());
    }

    // ---- Declaration tests ----

    #[test]
    fn test_declaration_name() {
        let decl = Declaration::Let(LetDecl {
            name: "x".into(),
            ty: None,
            value: spanned(Expr::int(42)),
            attributes: vec![],
            span: Span::synthetic(),
        });
        assert_eq!(decl.name(), Some("x"));
    }

    #[test]
    fn test_declaration_import_has_no_name() {
        let decl = Declaration::Import(ImportDecl {
            path: vec!["std".into(), "metrics".into()],
            alias: None,
            items: ImportItems::All,
            span: Span::synthetic(),
        });
        assert_eq!(decl.name(), None);
    }

    #[test]
    fn test_declaration_is_metric() {
        let decl = Declaration::Metric(MetricDecl {
            name: "exact".into(),
            params: vec![],
            return_type: EvalType::Semiring(SemiringType::Boolean),
            body: spanned(Expr::bool_lit(true)),
            attributes: vec![],
            metadata: MetricMetadata::empty(),
            span: Span::synthetic(),
        });
        assert!(decl.is_metric());
        assert!(!decl.is_test());
    }

    #[test]
    fn test_declaration_is_test() {
        let decl = Declaration::Test(TestDecl {
            name: "test_foo".into(),
            body: spanned(Expr::int(1)),
            expected: TestExpectation::Value(Literal::Integer(1)),
            span: Span::synthetic(),
        });
        assert!(decl.is_test());
        assert!(!decl.is_metric());
    }

    // ---- Program tests ----

    #[test]
    fn test_program_empty() {
        let p = Program::empty();
        assert!(p.declarations.is_empty());
        assert_eq!(p.declared_names().len(), 0);
    }

    #[test]
    fn test_program_metrics_iterator() {
        let program = Program::new(
            vec![
                spanned(Declaration::Metric(MetricDecl {
                    name: "m1".into(),
                    params: vec![],
                    return_type: EvalType::Semiring(SemiringType::Real),
                    body: spanned(Expr::float(1.0)),
                    attributes: vec![],
                    metadata: MetricMetadata::empty(),
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Let(LetDecl {
                    name: "x".into(),
                    ty: None,
                    value: spanned(Expr::int(1)),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Metric(MetricDecl {
                    name: "m2".into(),
                    params: vec![],
                    return_type: EvalType::Semiring(SemiringType::Boolean),
                    body: spanned(Expr::bool_lit(false)),
                    attributes: vec![],
                    metadata: MetricMetadata::empty(),
                    span: Span::synthetic(),
                })),
            ],
            Span::synthetic(),
        );
        let metrics: Vec<_> = program.metrics().collect();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].name, "m1");
        assert_eq!(metrics[1].name, "m2");
    }

    #[test]
    fn test_program_declared_names() {
        let program = Program::new(
            vec![
                spanned(Declaration::Let(LetDecl {
                    name: "a".into(),
                    ty: None,
                    value: spanned(Expr::int(1)),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Type(TypeDecl {
                    name: "MyType".into(),
                    ty: EvalType::Base(BaseType::Integer),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
            ],
            Span::synthetic(),
        );
        let names = program.declared_names();
        assert_eq!(names, vec!["a", "MyType"]);
    }

    #[test]
    fn test_program_tests_iterator() {
        let program = Program::new(
            vec![spanned(Declaration::Test(TestDecl {
                name: "t1".into(),
                body: spanned(Expr::int(1)),
                expected: TestExpectation::Success,
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        assert_eq!(program.tests().count(), 1);
    }

    #[test]
    fn test_program_imports_iterator() {
        let program = Program::new(
            vec![spanned(Declaration::Import(ImportDecl {
                path: vec!["std".into()],
                alias: None,
                items: ImportItems::All,
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        assert_eq!(program.imports().count(), 1);
    }

    #[test]
    fn test_program_let_decls_iterator() {
        let program = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "x".into(),
                ty: Some(EvalType::Base(BaseType::Integer)),
                value: spanned(Expr::int(42)),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let lets: Vec<_> = program.let_decls().collect();
        assert_eq!(lets.len(), 1);
        assert_eq!(lets[0].name, "x");
    }

    // ---- MetricType tests ----

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::ExactMatch.to_string(), "ExactMatch");
        assert_eq!(MetricType::BLEU.to_string(), "BLEU");
        assert_eq!(MetricType::RougeN.to_string(), "ROUGE-N");
        assert_eq!(MetricType::PassAtK.to_string(), "Pass@K");
        assert_eq!(MetricType::Custom.to_string(), "Custom");
    }

    // ---- NGramOrder tests ----

    #[test]
    fn test_ngram_order_valid() {
        let o = NGramOrder::new(3);
        assert!(o.is_some());
        assert_eq!(o.unwrap().value(), 3);
    }

    #[test]
    fn test_ngram_order_zero_invalid() {
        assert!(NGramOrder::new(0).is_none());
    }

    #[test]
    fn test_ngram_order_display() {
        assert_eq!(NGramOrder::new(4).unwrap().to_string(), "4");
    }

    // ---- AggregationOp tests ----

    #[test]
    fn test_aggregation_op_identity() {
        assert_eq!(AggregationOp::Sum.identity(), 0.0);
        assert_eq!(AggregationOp::Product.identity(), 1.0);
        assert_eq!(AggregationOp::Min.identity(), f64::INFINITY);
        assert_eq!(AggregationOp::Max.identity(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_aggregation_op_is_associative() {
        assert!(AggregationOp::Sum.is_associative());
        assert!(AggregationOp::Product.is_associative());
        assert!(AggregationOp::Min.is_associative());
        assert!(!AggregationOp::Mean.is_associative());
        assert!(!AggregationOp::HarmonicMean.is_associative());
    }

    #[test]
    fn test_aggregation_op_display() {
        assert_eq!(AggregationOp::Sum.to_string(), "sum");
        assert_eq!(AggregationOp::HarmonicMean.to_string(), "harmonic_mean");
        assert_eq!(AggregationOp::GeometricMean.to_string(), "geometric_mean");
        assert_eq!(AggregationOp::Count.to_string(), "count");
    }

    // ---- SmoothingMethod tests ----

    #[test]
    fn test_smoothing_default() {
        let s = SmoothingMethod::default();
        assert_eq!(s, SmoothingMethod::None);
    }

    #[test]
    fn test_smoothing_display() {
        assert_eq!(SmoothingMethod::None.to_string(), "none");
        assert_eq!(
            SmoothingMethod::AddK(OrderedFloat(1.0)).to_string(),
            "add-k(1)"
        );
        assert_eq!(SmoothingMethod::ChenCherry.to_string(), "chen-cherry");
        assert_eq!(SmoothingMethod::NIST.to_string(), "nist");
    }

    // ---- BLEUConfig tests ----

    #[test]
    fn test_bleu_config_default() {
        let c = BLEUConfig::default();
        assert_eq!(c.max_n, 4);
        assert!(c.brevity_penalty);
        assert_eq!(c.weights.len(), 4);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_bleu_config_invalid_zero_n() {
        let c = BLEUConfig {
            max_n: 0,
            ..BLEUConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_bleu_config_invalid_weights_length() {
        let c = BLEUConfig {
            max_n: 4,
            weights: vec![OrderedFloat(0.5), OrderedFloat(0.5)],
            ..BLEUConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_bleu_config_invalid_weights_sum() {
        let c = BLEUConfig {
            max_n: 2,
            weights: vec![OrderedFloat(0.3), OrderedFloat(0.3)],
            smoothing: SmoothingMethod::None,
            brevity_penalty: true,
        };
        assert!(c.validate().is_err());
    }

    // ---- RougeConfig tests ----

    #[test]
    fn test_rouge_config_default() {
        let c = RougeConfig::default();
        assert_eq!(c.n_gram_size, 1);
        assert!(!c.use_stemmer);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_rouge_config_invalid_zero() {
        let c = RougeConfig {
            n_gram_size: 0,
            ..RougeConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_rouge_scoring_type_display() {
        assert_eq!(RougeScoringType::Precision.to_string(), "precision");
        assert_eq!(RougeScoringType::Recall.to_string(), "recall");
        assert_eq!(RougeScoringType::FMeasure.to_string(), "f-measure");
    }

    // ---- PassAtKConfig tests ----

    #[test]
    fn test_pass_at_k_default() {
        let c = PassAtKConfig::default();
        assert_eq!(c.k_values, vec![1, 10, 100]);
        assert_eq!(c.num_samples, 200);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_pass_at_k_invalid_empty() {
        let c = PassAtKConfig {
            k_values: vec![],
            num_samples: 10,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_pass_at_k_invalid_zero_k() {
        let c = PassAtKConfig {
            k_values: vec![0],
            num_samples: 10,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_pass_at_k_invalid_k_exceeds_samples() {
        let c = PassAtKConfig {
            k_values: vec![100],
            num_samples: 10,
        };
        assert!(c.validate().is_err());
    }

    // ---- TypeEnv tests ----

    #[test]
    fn test_type_env_basic() {
        let mut env = TypeEnv::new();
        assert_eq!(env.depth(), 1);
        env.bind("x", EvalType::Base(BaseType::Integer), false);
        assert!(env.contains("x"));
        assert!(!env.contains("y"));
        let binding = env.lookup("x").unwrap();
        assert_eq!(binding.ty, EvalType::Base(BaseType::Integer));
        assert!(!binding.mutable);
    }

    #[test]
    fn test_type_env_scoping() {
        let mut env = TypeEnv::new();
        env.bind("x", EvalType::Base(BaseType::Integer), false);
        env.push_scope();
        env.bind("x", EvalType::Base(BaseType::Float), true);
        // Inner scope shadows outer.
        let binding = env.lookup("x").unwrap();
        assert_eq!(binding.ty, EvalType::Base(BaseType::Float));
        assert!(binding.mutable);
        env.pop_scope();
        // Outer scope restored.
        let binding = env.lookup("x").unwrap();
        assert_eq!(binding.ty, EvalType::Base(BaseType::Integer));
    }

    #[test]
    fn test_type_env_all_bindings() {
        let mut env = TypeEnv::new();
        env.bind("a", EvalType::Base(BaseType::Integer), false);
        env.push_scope();
        env.bind("b", EvalType::Base(BaseType::Bool), false);
        let all = env.all_bindings();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("a"));
        assert!(all.contains_key("b"));
    }

    #[test]
    fn test_type_env_current_scope_names() {
        let mut env = TypeEnv::new();
        env.bind("x", EvalType::Base(BaseType::Integer), false);
        env.push_scope();
        env.bind("y", EvalType::Base(BaseType::Bool), false);
        let names = env.current_scope_names();
        assert_eq!(names, vec!["y"]);
    }

    #[test]
    fn test_type_env_cannot_pop_last_scope() {
        let mut env = TypeEnv::new();
        env.pop_scope(); // should not panic
        assert_eq!(env.depth(), 1);
    }

    // ---- Substitution tests ----

    #[test]
    fn test_substitution_empty() {
        let s = Substitution::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_substitution_singleton() {
        let s = Substitution::singleton("a", EvalType::Base(BaseType::Integer));
        assert_eq!(s.len(), 1);
        assert_eq!(
            s.lookup("a"),
            Some(&EvalType::Base(BaseType::Integer))
        );
    }

    #[test]
    fn test_substitution_compose() {
        let mut s1 = Substitution::empty();
        s1.insert("a", EvalType::TypeVar("b".into()));
        let mut s2 = Substitution::empty();
        s2.insert("b", EvalType::Base(BaseType::Integer));
        let composed = s2.compose(&s1);
        // s1 applied first: a -> b, then s2: b -> Int. So a -> Int.
        assert_eq!(
            composed.lookup("a"),
            Some(&EvalType::Base(BaseType::Integer))
        );
        assert_eq!(
            composed.lookup("b"),
            Some(&EvalType::Base(BaseType::Integer))
        );
    }

    // ---- TypeConstraint tests ----

    #[test]
    fn test_type_constraint_span() {
        let s = span(1, 0);
        let c = TypeConstraint::Equal(
            EvalType::Base(BaseType::Integer),
            EvalType::Base(BaseType::Float),
            s.clone(),
        );
        assert_eq!(c.span(), &s);
    }

    #[test]
    fn test_type_constraint_is_callable() {
        let s = span(5, 0);
        let c = TypeConstraint::IsCallable {
            ty: EvalType::TypeVar("f".into()),
            args: vec![EvalType::Base(BaseType::Integer)],
            ret: EvalType::Base(BaseType::Bool),
            span: s.clone(),
        };
        assert_eq!(c.span(), &s);
    }

    // ---- MatchMode tests ----

    #[test]
    fn test_match_mode_display() {
        assert_eq!(MatchMode::Regex.to_string(), "regex");
        assert_eq!(MatchMode::Glob.to_string(), "glob");
        assert_eq!(MatchMode::Exact.to_string(), "exact");
        assert_eq!(MatchMode::Contains.to_string(), "contains");
    }

    // ---- Validation tests ----

    #[test]
    fn test_validate_empty_program() {
        let p = Program::empty();
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::EmptyProgram)));
    }

    #[test]
    fn test_validate_valid_program() {
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "x".into(),
                ty: None,
                value: spanned(Expr::int(1)),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        assert!(validate_program(&p).is_ok());
    }

    #[test]
    fn test_validate_duplicate_names() {
        let s1 = Span::new("test.eval", 1, 0, 1, 5);
        let s2 = Span::new("test.eval", 2, 0, 2, 5);
        let p = Program::new(
            vec![
                Spanned::new(
                    Declaration::Let(LetDecl {
                        name: "x".into(),
                        ty: None,
                        value: spanned(Expr::int(1)),
                        attributes: vec![],
                        span: s1.clone(),
                    }),
                    s1,
                ),
                Spanned::new(
                    Declaration::Let(LetDecl {
                        name: "x".into(),
                        ty: None,
                        value: spanned(Expr::int(2)),
                        attributes: vec![],
                        span: s2.clone(),
                    }),
                    s2,
                ),
            ],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateName { .. })));
    }

    #[test]
    fn test_validate_empty_block() {
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "x".into(),
                ty: None,
                value: spanned(Expr::Block(vec![])),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::EmptyBlock { .. })));
    }

    #[test]
    fn test_validate_invalid_ngram_order() {
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "ng".into(),
                ty: None,
                value: spanned(Expr::NGramExtract {
                    input: Box::new(spanned(Expr::var("tokens"))),
                    n: 0,
                }),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidNGramOrder { .. })));
    }

    #[test]
    fn test_validate_empty_match() {
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "m".into(),
                ty: None,
                value: spanned(Expr::Match {
                    scrutinee: Box::new(spanned(Expr::int(1))),
                    arms: vec![],
                }),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::NonExhaustiveMatch { .. })));
    }

    #[test]
    fn test_validate_duplicate_params() {
        let p = Program::new(
            vec![spanned(Declaration::Metric(MetricDecl {
                name: "bad_metric".into(),
                params: vec![
                    MetricParameter {
                        name: "x".into(),
                        ty: EvalType::Base(BaseType::Integer),
                        default: None,
                        span: Span::synthetic(),
                    },
                    MetricParameter {
                        name: "x".into(),
                        ty: EvalType::Base(BaseType::Float),
                        default: None,
                        span: Span::synthetic(),
                    },
                ],
                return_type: EvalType::Semiring(SemiringType::Real),
                body: spanned(Expr::float(0.0)),
                attributes: vec![],
                metadata: MetricMetadata::empty(),
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateParameter { .. })));
    }

    #[test]
    fn test_validate_empty_import_path() {
        let p = Program::new(
            vec![spanned(Declaration::Import(ImportDecl {
                path: vec![],
                alias: None,
                items: ImportItems::All,
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let result = validate_program(&p);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::ImportNotFound { .. })));
    }

    #[test]
    fn test_validate_nested_valid_exprs() {
        let body = Expr::If {
            condition: Box::new(spanned(Expr::bool_lit(true))),
            then_branch: Box::new(spanned(Expr::binary(
                BinaryOp::Add,
                spanned(Expr::int(1)),
                spanned(Expr::int(2)),
            ))),
            else_branch: Box::new(spanned(Expr::int(0))),
        };
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "r".into(),
                ty: None,
                value: spanned(body),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        assert!(validate_program(&p).is_ok());
    }

    // ---- Serialization round-trip tests ----

    #[test]
    fn test_serde_roundtrip_semiring_type() {
        for sr in [
            SemiringType::Counting,
            SemiringType::Boolean,
            SemiringType::Tropical,
            SemiringType::BoundedCounting(100),
            SemiringType::Real,
            SemiringType::LogDomain,
            SemiringType::Viterbi,
            SemiringType::Goldilocks,
        ] {
            let json = serde_json::to_string(&sr).unwrap();
            let deser: SemiringType = serde_json::from_str(&json).unwrap();
            assert_eq!(sr, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_base_type() {
        let types = vec![
            BaseType::String,
            BaseType::Integer,
            BaseType::Float,
            BaseType::Bool,
            BaseType::Token,
            BaseType::TokenSequence,
            BaseType::NGram(3),
            BaseType::List(Box::new(BaseType::Integer)),
            BaseType::Tuple(vec![BaseType::String, BaseType::Bool]),
        ];
        for ty in types {
            let json = serde_json::to_string(&ty).unwrap();
            let deser: BaseType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_eval_type() {
        let types = vec![
            EvalType::Base(BaseType::Integer),
            EvalType::Semiring(SemiringType::Real),
            EvalType::Unit,
            EvalType::TypeVar("alpha".into()),
            EvalType::Function {
                params: vec![EvalType::Base(BaseType::String)],
                ret: Box::new(EvalType::Base(BaseType::Bool)),
            },
            EvalType::Metric {
                input: Box::new(EvalType::Base(BaseType::String)),
                output: Box::new(EvalType::Semiring(SemiringType::Real)),
            },
            EvalType::Annotated {
                base: BaseType::Float,
                semiring: SemiringType::LogDomain,
            },
        ];
        for ty in types {
            let json = serde_json::to_string(&ty).unwrap();
            let deser: EvalType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_literal() {
        let lits = vec![
            Literal::String("hello".into()),
            Literal::Integer(42),
            Literal::Float(OrderedFloat(3.14)),
            Literal::Bool(true),
        ];
        for lit in lits {
            let json = serde_json::to_string(&lit).unwrap();
            let deser: Literal = serde_json::from_str(&json).unwrap();
            assert_eq!(lit, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_expr() {
        let exprs = vec![
            Expr::int(42),
            Expr::var("x"),
            Expr::bool_lit(true),
            Expr::string("hello"),
            Expr::float(2.718),
            Expr::call("f", vec![spanned(Expr::int(1))]),
            Expr::ListLiteral(vec![spanned(Expr::int(1)), spanned(Expr::int(2))]),
        ];
        for expr in exprs {
            let json = serde_json::to_string(&expr).unwrap();
            let deser: Expr = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_pattern() {
        let patterns = vec![
            Pattern::Wildcard,
            Pattern::Var("x".into()),
            Pattern::Literal(Literal::Integer(1)),
            Pattern::Tuple(vec![
                spanned(Pattern::Var("a".into())),
                spanned(Pattern::Wildcard),
            ]),
        ];
        for pat in patterns {
            let json = serde_json::to_string(&pat).unwrap();
            let deser: Pattern = serde_json::from_str(&json).unwrap();
            assert_eq!(pat, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_span() {
        let s = Span::new("myfile.eval", 10, 5, 10, 20);
        let json = serde_json::to_string(&s).unwrap();
        let deser: Span = serde_json::from_str(&json).unwrap();
        assert_eq!(s, deser);
    }

    #[test]
    fn test_serde_roundtrip_binary_op() {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Min,
            BinaryOp::Max,
            BinaryOp::And,
            BinaryOp::Or,
            BinaryOp::Eq,
            BinaryOp::Neq,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
        ] {
            let json = serde_json::to_string(&op).unwrap();
            let deser: BinaryOp = serde_json::from_str(&json).unwrap();
            assert_eq!(op, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_bleu_config() {
        let c = BLEUConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let deser: BLEUConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, deser);
    }

    #[test]
    fn test_serde_roundtrip_rouge_config() {
        let c = RougeConfig {
            n_gram_size: 2,
            use_stemmer: true,
            stopwords: vec!["the".into(), "a".into()],
            scoring_type: RougeScoringType::Recall,
        };
        let json = serde_json::to_string(&c).unwrap();
        let deser: RougeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, deser);
    }

    #[test]
    fn test_serde_roundtrip_pass_at_k_config() {
        let c = PassAtKConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let deser: PassAtKConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, deser);
    }

    #[test]
    fn test_serde_roundtrip_attribute() {
        let attrs = vec![
            Attribute::Doc("A test".into()),
            Attribute::Deprecated("old".into()),
            Attribute::Test,
            Attribute::Inline,
            Attribute::SemiringHint(SemiringType::Boolean),
        ];
        for attr in attrs {
            let json = serde_json::to_string(&attr).unwrap();
            let deser: Attribute = serde_json::from_str(&json).unwrap();
            assert_eq!(attr, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_metric_metadata() {
        let m = MetricMetadata {
            author: Some("Alice".into()),
            version: Some("1.0".into()),
            description: Some("A cool metric".into()),
            tags: vec!["nlp".into(), "evaluation".into()],
            created_at: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deser: MetricMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deser);
    }

    #[test]
    fn test_serde_roundtrip_declaration() {
        let decl = Declaration::Let(LetDecl {
            name: "x".into(),
            ty: Some(EvalType::Base(BaseType::Integer)),
            value: spanned(Expr::int(42)),
            attributes: vec![Attribute::Doc("the answer".into())],
            span: Span::synthetic(),
        });
        let json = serde_json::to_string(&decl).unwrap();
        let deser: Declaration = serde_json::from_str(&json).unwrap();
        assert_eq!(decl, deser);
    }

    #[test]
    fn test_serde_roundtrip_metric_spec() {
        let spec = MetricSpec {
            name: "my_metric".into(),
            metric_type: MetricType::Custom,
            params: vec![],
            semiring: SemiringType::Real,
            body: spanned(Expr::float(1.0)),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let deser: MetricSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, deser);
    }

    #[test]
    fn test_serde_roundtrip_test_expectation() {
        let expectations = vec![
            TestExpectation::Success,
            TestExpectation::Value(Literal::Integer(42)),
            TestExpectation::Approx {
                value: OrderedFloat(0.5),
                tolerance: OrderedFloat(0.01),
            },
            TestExpectation::Error(Some("division by zero".into())),
            TestExpectation::Error(None),
        ];
        for exp in expectations {
            let json = serde_json::to_string(&exp).unwrap();
            let deser: TestExpectation = serde_json::from_str(&json).unwrap();
            assert_eq!(exp, deser);
        }
    }

    #[test]
    fn test_serde_roundtrip_import_items() {
        let items = ImportItems::Named(vec![
            ImportItem {
                name: "foo".into(),
                alias: None,
            },
            ImportItem {
                name: "bar".into(),
                alias: Some("baz".into()),
            },
        ]);
        let json = serde_json::to_string(&items).unwrap();
        let deser: ImportItems = serde_json::from_str(&json).unwrap();
        assert_eq!(items, deser);
    }

    // ---- Program Display tests ----

    #[test]
    fn test_program_display_let() {
        let p = Program::new(
            vec![spanned(Declaration::Let(LetDecl {
                name: "x".into(),
                ty: Some(EvalType::Base(BaseType::Integer)),
                value: spanned(Expr::int(42)),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let output = p.to_string();
        assert!(output.contains("let x: Int = 42"));
    }

    #[test]
    fn test_program_display_metric() {
        let p = Program::new(
            vec![spanned(Declaration::Metric(MetricDecl {
                name: "exact_match".into(),
                params: vec![MetricParameter {
                    name: "ref".into(),
                    ty: EvalType::Base(BaseType::String),
                    default: None,
                    span: Span::synthetic(),
                }],
                return_type: EvalType::Semiring(SemiringType::Boolean),
                body: spanned(Expr::bool_lit(true)),
                attributes: vec![Attribute::Doc("Checks exact match".into())],
                metadata: MetricMetadata::empty(),
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let output = p.to_string();
        assert!(output.contains("metric exact_match"));
        assert!(output.contains("#[doc"));
    }

    #[test]
    fn test_program_display_import_all() {
        let p = Program::new(
            vec![spanned(Declaration::Import(ImportDecl {
                path: vec!["std".into(), "metrics".into()],
                alias: None,
                items: ImportItems::All,
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let output = p.to_string();
        assert!(output.contains("import std::metrics::*"));
    }

    #[test]
    fn test_program_display_import_named() {
        let p = Program::new(
            vec![spanned(Declaration::Import(ImportDecl {
                path: vec!["std".into()],
                alias: None,
                items: ImportItems::Named(vec![
                    ImportItem {
                        name: "bleu".into(),
                        alias: None,
                    },
                    ImportItem {
                        name: "rouge".into(),
                        alias: Some("rg".into()),
                    },
                ]),
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let output = p.to_string();
        assert!(output.contains("bleu"));
        assert!(output.contains("rouge as rg"));
    }

    #[test]
    fn test_program_display_type_decl() {
        let p = Program::new(
            vec![spanned(Declaration::Type(TypeDecl {
                name: "Score".into(),
                ty: EvalType::Semiring(SemiringType::Real),
                attributes: vec![],
                span: Span::synthetic(),
            }))],
            Span::synthetic(),
        );
        let output = p.to_string();
        assert!(output.contains("type Score = Semiring<Real>"));
    }

    // ---- ValidationError tests ----

    #[test]
    fn test_validation_error_span() {
        let s = span(1, 0);
        let e = ValidationError::UndefinedVariable {
            name: "foo".into(),
            span: s.clone(),
        };
        assert_eq!(e.span(), Some(&s));
    }

    #[test]
    fn test_validation_error_empty_program_no_span() {
        assert_eq!(ValidationError::EmptyProgram.span(), None);
    }

    #[test]
    fn test_validation_error_display() {
        let e = ValidationError::TypeMismatch {
            expected: "Int".into(),
            found: "Bool".into(),
            span: span(3, 5),
        };
        let msg = e.to_string();
        assert!(msg.contains("type mismatch"));
        assert!(msg.contains("Int"));
        assert!(msg.contains("Bool"));
    }

    // ---- MatchArm construction test ----

    #[test]
    fn test_match_arm_construction() {
        let arm = MatchArm {
            pattern: spanned(Pattern::Literal(Literal::Integer(0))),
            guard: Some(spanned(Expr::bool_lit(true))),
            body: spanned(Expr::string("zero")),
            span: Span::synthetic(),
        };
        assert_eq!(arm.pattern.node, Pattern::Literal(Literal::Integer(0)));
        assert!(arm.guard.is_some());
    }

    // ---- Complex expression tree test ----

    #[test]
    fn test_complex_expression_tree() {
        // Build: let score = sum(x in tokens => clip(count(x, ref), count(x, hyp))) in score / len(tokens)
        let inner_clip = Expr::ClipCount {
            count: Box::new(spanned(Expr::call(
                "count",
                vec![spanned(Expr::var("x")), spanned(Expr::var("ref"))],
            ))),
            max_count: Box::new(spanned(Expr::call(
                "count",
                vec![spanned(Expr::var("x")), spanned(Expr::var("hyp"))],
            ))),
        };
        let agg = Expr::Aggregate {
            op: AggregationOp::Sum,
            collection: Box::new(spanned(Expr::var("tokens"))),
            binding: Some("x".into()),
            body: Some(Box::new(spanned(inner_clip))),
            semiring: Some(SemiringType::Counting),
        };
        let full = Expr::let_expr(
            "score",
            None,
            spanned(agg),
            spanned(Expr::binary(
                BinaryOp::Div,
                spanned(Expr::var("score")),
                spanned(Expr::call("len", vec![spanned(Expr::var("tokens"))])),
            )),
        );
        assert!(full.node_count() > 10);
        assert!(full.depth() > 3);
    }

    // ---- MetricSpec construction test ----

    #[test]
    fn test_metric_spec_construction() {
        let spec = MetricSpec {
            name: "token_f1".into(),
            metric_type: MetricType::TokenF1,
            params: vec![
                MetricParameter {
                    name: "reference".into(),
                    ty: EvalType::Base(BaseType::TokenSequence),
                    default: None,
                    span: Span::synthetic(),
                },
                MetricParameter {
                    name: "candidate".into(),
                    ty: EvalType::Base(BaseType::TokenSequence),
                    default: None,
                    span: Span::synthetic(),
                },
            ],
            semiring: SemiringType::Real,
            body: spanned(Expr::float(0.0)),
        };
        assert_eq!(spec.name, "token_f1");
        assert_eq!(spec.metric_type, MetricType::TokenF1);
        assert_eq!(spec.params.len(), 2);
    }

    // ---- Full program validation test ----

    #[test]
    fn test_full_valid_program() {
        let program = Program::new(
            vec![
                spanned(Declaration::Import(ImportDecl {
                    path: vec!["std".into(), "metrics".into()],
                    alias: None,
                    items: ImportItems::All,
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Type(TypeDecl {
                    name: "Score".into(),
                    ty: EvalType::Semiring(SemiringType::Real),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Let(LetDecl {
                    name: "threshold".into(),
                    ty: Some(EvalType::Base(BaseType::Float)),
                    value: spanned(Expr::float(0.5)),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Metric(MetricDecl {
                    name: "my_metric".into(),
                    params: vec![
                        MetricParameter {
                            name: "ref_text".into(),
                            ty: EvalType::Base(BaseType::String),
                            default: None,
                            span: Span::synthetic(),
                        },
                        MetricParameter {
                            name: "hyp_text".into(),
                            ty: EvalType::Base(BaseType::String),
                            default: None,
                            span: Span::synthetic(),
                        },
                    ],
                    return_type: EvalType::Semiring(SemiringType::Real),
                    body: spanned(Expr::if_expr(
                        spanned(Expr::binary(
                            BinaryOp::Eq,
                            spanned(Expr::var("ref_text")),
                            spanned(Expr::var("hyp_text")),
                        )),
                        spanned(Expr::float(1.0)),
                        spanned(Expr::float(0.0)),
                    )),
                    attributes: vec![Attribute::Doc("My custom metric".into())],
                    metadata: MetricMetadata {
                        author: Some("Test Author".into()),
                        version: Some("0.1.0".into()),
                        description: Some("A test metric".into()),
                        tags: vec!["test".into()],
                        created_at: None,
                    },
                    span: Span::synthetic(),
                })),
                spanned(Declaration::Test(TestDecl {
                    name: "test_exact".into(),
                    body: spanned(Expr::call(
                        "my_metric",
                        vec![
                            spanned(Expr::string("hello")),
                            spanned(Expr::string("hello")),
                        ],
                    )),
                    expected: TestExpectation::Approx {
                        value: OrderedFloat(1.0),
                        tolerance: OrderedFloat(1e-6),
                    },
                    span: Span::synthetic(),
                })),
            ],
            Span::synthetic(),
        );
        assert!(validate_program(&program).is_ok());
        assert_eq!(program.metrics().count(), 1);
        assert_eq!(program.tests().count(), 1);
        assert_eq!(program.imports().count(), 1);
        assert_eq!(program.type_decls().count(), 1);
        assert_eq!(program.let_decls().count(), 1);
        assert_eq!(program.declared_names().len(), 4); // Score, threshold, my_metric, test_exact
    }
}
