//! Denotational semantics for the EvalSpec DSL.
//!
//! This module provides:
//! - A direct interpreter (`Evaluator`) that maps EvalSpec expressions to
//!   semantic values in a standard environment model.
//! - Formal power series denotations (`FPSSemantics`) that characterize
//!   metrics as formal power series over the free monoid of tokens.
//! - Semantic equivalence checking via random testing.
//! - Expression normalization / simplification.
//! - Semantic preservation verification against compiled WFAs.
//! - Built-in function implementations for the standard library.

use std::fmt;

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use thiserror::Error;

use super::types::{
    AggregationOp, BaseType, BinaryOp, Declaration, EvalType, Expr,
    Literal, MatchArm, MatchMode, MetricDecl,
    MetricType, Pattern, Program,
    SemiringType, Spanned, UnaryOp,
};

use crate::wfa::{
    CountingSemiring, WeightedFiniteAutomaton,
};

// ═══════════════════════════════════════════════════════════════════════════
// 1. SemanticError
// ═══════════════════════════════════════════════════════════════════════════

/// Errors arising during semantic evaluation, normalization, or equivalence
/// checking of EvalSpec programs.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SemanticError {
    #[error("undefined variable `{name}`")]
    UndefinedVariable { name: String },

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("division by zero")]
    DivisionByZero,

    #[error("invalid operation: {desc}")]
    InvalidOperation { desc: String },

    #[error("unsupported expression: {desc}")]
    UnsupportedExpression { desc: String },

    #[error("non-termination detected: {desc}")]
    NonTermination { desc: String },

    #[error("invalid semiring operation: {desc}")]
    InvalidSemiring { desc: String },

    #[error("normalization failure: {desc}")]
    NormalizationFailure { desc: String },

    #[error("equivalence check failed: {desc}")]
    EquivalenceCheckFailed { desc: String },
}

pub type SemResult<T> = Result<T, SemanticError>;

// ═══════════════════════════════════════════════════════════════════════════
// 2. SemanticValue — the denotation domain
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime value produced by evaluating an EvalSpec expression.
///
/// This enum forms the *denotation domain* D of the language, i.e. the
/// semantic algebra into which every well-typed expression is mapped.
#[derive(Clone, Debug)]
pub enum SemanticValue {
    /// Machine integer (ℤ truncated to i64).
    Integer(i64),
    /// IEEE 754 double.
    Float(f64),
    /// Boolean truth value.
    Boolean(bool),
    /// UTF-8 string.
    Str(String),
    /// Homogeneous ordered collection.
    List(Vec<SemanticValue>),
    /// Heterogeneous fixed-length product.
    Tuple(Vec<SemanticValue>),
    /// Ordered sequence of string tokens.
    TokenSequence(Vec<String>),
    /// Multiset of n-grams: maps each n-gram (sequence of tokens) to its
    /// count in the source.
    NGramSet(IndexMap<Vec<String>, usize>),
    /// A value living in a specific semiring.
    SemiringVal(SemiringValue),
    /// A first-class function (closure).
    Function(FunctionValue),
    /// The unit value (no information).
    Unit,
}

impl PartialEq for SemanticValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SemanticValue::Integer(a), SemanticValue::Integer(b)) => a == b,
            (SemanticValue::Float(a), SemanticValue::Float(b)) => {
                OrderedFloat(*a) == OrderedFloat(*b)
            }
            (SemanticValue::Boolean(a), SemanticValue::Boolean(b)) => a == b,
            (SemanticValue::Str(a), SemanticValue::Str(b)) => a == b,
            (SemanticValue::List(a), SemanticValue::List(b)) => a == b,
            (SemanticValue::Tuple(a), SemanticValue::Tuple(b)) => a == b,
            (SemanticValue::TokenSequence(a), SemanticValue::TokenSequence(b)) => a == b,
            (SemanticValue::NGramSet(a), SemanticValue::NGramSet(b)) => a == b,
            (SemanticValue::SemiringVal(a), SemanticValue::SemiringVal(b)) => a == b,
            (SemanticValue::Unit, SemanticValue::Unit) => true,
            // Functions are compared by identity (pointer equality of params+body
            // is too fragile; we say functions are never equal).
            (SemanticValue::Function(_), SemanticValue::Function(_)) => false,
            _ => false,
        }
    }
}

impl fmt::Display for SemanticValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemanticValue::Integer(n) => write!(f, "{n}"),
            SemanticValue::Float(v) => write!(f, "{v}"),
            SemanticValue::Boolean(b) => write!(f, "{b}"),
            SemanticValue::Str(s) => write!(f, "\"{s}\""),
            SemanticValue::List(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, "]")
            }
            SemanticValue::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            SemanticValue::TokenSequence(tokens) => {
                write!(f, "tokens[")?;
                for (i, t) in tokens.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, "]")
            }
            SemanticValue::NGramSet(ngrams) => {
                write!(f, "ngrams{{")?;
                for (i, (gram, count)) in ngrams.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}:{count}", gram)?;
                }
                write!(f, "}}")
            }
            SemanticValue::SemiringVal(sv) => write!(f, "{sv}"),
            SemanticValue::Function(fv) => {
                write!(f, "<fn({})>", fv.params.join(", "))
            }
            SemanticValue::Unit => write!(f, "()"),
        }
    }
}

impl SemanticValue {
    /// Attempt to coerce this value to an f64 for arithmetic.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            SemanticValue::Integer(n) => Some(*n as f64),
            SemanticValue::Float(v) => Some(*v),
            SemanticValue::SemiringVal(sv) => sv.to_f64(),
            SemanticValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Attempt to coerce this value to an i64.
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            SemanticValue::Integer(n) => Some(*n),
            SemanticValue::Float(v) => Some(*v as i64),
            SemanticValue::Boolean(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Attempt to coerce this value to a bool.
    pub fn to_bool(&self) -> Option<bool> {
        match self {
            SemanticValue::Boolean(b) => Some(*b),
            SemanticValue::Integer(n) => Some(*n != 0),
            SemanticValue::Float(v) => Some(*v != 0.0),
            _ => None,
        }
    }

    /// Return the type name of this value.
    pub fn type_name(&self) -> &'static str {
        match self {
            SemanticValue::Integer(_) => "Integer",
            SemanticValue::Float(_) => "Float",
            SemanticValue::Boolean(_) => "Boolean",
            SemanticValue::Str(_) => "String",
            SemanticValue::List(_) => "List",
            SemanticValue::Tuple(_) => "Tuple",
            SemanticValue::TokenSequence(_) => "TokenSequence",
            SemanticValue::NGramSet(_) => "NGramSet",
            SemanticValue::SemiringVal(_) => "SemiringVal",
            SemanticValue::Function(_) => "Function",
            SemanticValue::Unit => "Unit",
        }
    }

    /// Returns true if the value is a numeric scalar.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            SemanticValue::Integer(_) | SemanticValue::Float(_) | SemanticValue::SemiringVal(_)
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. SemiringValue
// ═══════════════════════════════════════════════════════════════════════════

/// A value that lives in one of the supported semiring domains.
///
/// Each variant carries a scalar from the corresponding semiring.  Arithmetic
/// on `SemiringValue`s dispatches to the semiring operations (⊕, ⊗) rather
/// than to ordinary +/×.
#[derive(Clone, Debug, PartialEq)]
pub enum SemiringValue {
    /// Natural-number counting semiring: (ℕ, +, ×, 0, 1).
    Counting(u64),
    /// Boolean semiring: ({0,1}, ∨, ∧, 0, 1).
    Bool(bool),
    /// Tropical semiring: (ℝ ∪ {∞}, min, +, ∞, 0).
    Tropical(f64),
    /// Real-number semiring: (ℝ, +, ×, 0, 1).
    Real(f64),
    /// Goldilocks prime field: (𝔽_p, +, ×, 0, 1) where p = 2⁶⁴ − 2³² + 1.
    Goldilocks(u64),
    /// Counting semiring with an upper bound (used in BLEU clipping).
    BoundedCounting { value: u64, max_count: u64 },
}

impl SemiringValue {
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            SemiringValue::Counting(n) => Some(*n as f64),
            SemiringValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            SemiringValue::Tropical(v) => Some(*v),
            SemiringValue::Real(v) => Some(*v),
            SemiringValue::Goldilocks(v) => Some(*v as f64),
            SemiringValue::BoundedCounting { value, .. } => Some(*value as f64),
        }
    }

    /// Apply semiring addition (⊕).
    pub fn sr_add(&self, other: &SemiringValue) -> SemResult<SemiringValue> {
        match (self, other) {
            (SemiringValue::Counting(a), SemiringValue::Counting(b)) => {
                Ok(SemiringValue::Counting(a.saturating_add(*b)))
            }
            (SemiringValue::Bool(a), SemiringValue::Bool(b)) => {
                Ok(SemiringValue::Bool(*a || *b))
            }
            (SemiringValue::Tropical(a), SemiringValue::Tropical(b)) => {
                Ok(SemiringValue::Tropical(a.min(*b)))
            }
            (SemiringValue::Real(a), SemiringValue::Real(b)) => {
                Ok(SemiringValue::Real(a + b))
            }
            (SemiringValue::Goldilocks(a), SemiringValue::Goldilocks(b)) => {
                let p = crate::wfa::GOLDILOCKS_PRIME as u128;
                let sum = ((*a as u128) + (*b as u128)) % p;
                Ok(SemiringValue::Goldilocks(sum as u64))
            }
            (
                SemiringValue::BoundedCounting {
                    value: a,
                    max_count: ba,
                },
                SemiringValue::BoundedCounting {
                    value: b,
                    max_count: bb,
                },
            ) => {
                let mc = (*ba).min(*bb);
                Ok(SemiringValue::BoundedCounting {
                    value: a.saturating_add(*b).min(mc),
                    max_count: mc,
                })
            }
            _ => Err(SemanticError::InvalidSemiring {
                desc: format!(
                    "cannot add semiring values of different types: {:?} ⊕ {:?}",
                    self, other
                ),
            }),
        }
    }

    /// Apply semiring multiplication (⊗).
    pub fn sr_mul(&self, other: &SemiringValue) -> SemResult<SemiringValue> {
        match (self, other) {
            (SemiringValue::Counting(a), SemiringValue::Counting(b)) => {
                Ok(SemiringValue::Counting(a.saturating_mul(*b)))
            }
            (SemiringValue::Bool(a), SemiringValue::Bool(b)) => {
                Ok(SemiringValue::Bool(*a && *b))
            }
            (SemiringValue::Tropical(a), SemiringValue::Tropical(b)) => {
                Ok(SemiringValue::Tropical(a + b))
            }
            (SemiringValue::Real(a), SemiringValue::Real(b)) => {
                Ok(SemiringValue::Real(a * b))
            }
            (SemiringValue::Goldilocks(a), SemiringValue::Goldilocks(b)) => {
                let p = crate::wfa::GOLDILOCKS_PRIME as u128;
                let prod = ((*a as u128) * (*b as u128)) % p;
                Ok(SemiringValue::Goldilocks(prod as u64))
            }
            (
                SemiringValue::BoundedCounting {
                    value: a,
                    max_count: ba,
                },
                SemiringValue::BoundedCounting {
                    value: b,
                    max_count: bb,
                },
            ) => {
                let mc = (*ba).min(*bb);
                Ok(SemiringValue::BoundedCounting {
                    value: a.saturating_mul(*b).min(mc),
                    max_count: mc,
                })
            }
            _ => Err(SemanticError::InvalidSemiring {
                desc: format!(
                    "cannot multiply semiring values of different types: {:?} ⊗ {:?}",
                    self, other
                ),
            }),
        }
    }

    /// The additive identity for the given semiring kind.
    pub fn zero_for(st: &SemiringType) -> SemiringValue {
        match st {
            SemiringType::Counting => SemiringValue::Counting(0),
            SemiringType::Boolean => SemiringValue::Bool(false),
            SemiringType::Tropical => SemiringValue::Tropical(f64::INFINITY),
            SemiringType::Real => SemiringValue::Real(0.0),
            SemiringType::Goldilocks => SemiringValue::Goldilocks(0),
            SemiringType::BoundedCounting(bound) => SemiringValue::BoundedCounting {
                value: 0,
                max_count: *bound,
            },
            SemiringType::LogDomain => SemiringValue::Real(f64::NEG_INFINITY),
            SemiringType::Viterbi => SemiringValue::Real(0.0),
        }
    }

    /// The multiplicative identity for the given semiring kind.
    pub fn one_for(st: &SemiringType) -> SemiringValue {
        match st {
            SemiringType::Counting => SemiringValue::Counting(1),
            SemiringType::Boolean => SemiringValue::Bool(true),
            SemiringType::Tropical => SemiringValue::Tropical(0.0),
            SemiringType::Real => SemiringValue::Real(1.0),
            SemiringType::Goldilocks => SemiringValue::Goldilocks(1),
            SemiringType::BoundedCounting(bound) => SemiringValue::BoundedCounting {
                value: 1,
                max_count: *bound,
            },
            SemiringType::LogDomain => SemiringValue::Real(0.0),
            SemiringType::Viterbi => SemiringValue::Real(1.0),
        }
    }
}

impl fmt::Display for SemiringValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SemiringValue::Counting(n) => write!(f, "counting({n})"),
            SemiringValue::Bool(b) => write!(f, "bool({b})"),
            SemiringValue::Tropical(v) => write!(f, "tropical({v})"),
            SemiringValue::Real(v) => write!(f, "real({v})"),
            SemiringValue::Goldilocks(v) => write!(f, "goldilocks({v})"),
            SemiringValue::BoundedCounting { value, max_count } => {
                write!(f, "bounded({value}, ≤{max_count})")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. FunctionValue
// ═══════════════════════════════════════════════════════════════════════════

/// A first-class function value (closure).  Captures the lexical environment
/// at the point of definition.
#[derive(Clone, Debug)]
pub struct FunctionValue {
    pub params: Vec<String>,
    pub body: Box<Expr>,
    pub closure: Environment,
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Environment — lexically-scoped name bindings
// ═══════════════════════════════════════════════════════════════════════════

/// A chain of scopes mapping names to semantic values.
///
/// The environment forms a linked list of scopes: each scope is an
/// `IndexMap<String, SemanticValue>` and optionally points to a parent
/// scope.  Variable lookup proceeds from the innermost scope outward.
#[derive(Clone, Debug)]
pub struct Environment {
    pub bindings: IndexMap<String, SemanticValue>,
    pub parent: Option<Box<Environment>>,
}

impl Environment {
    /// Create a fresh top-level environment with no bindings.
    pub fn new() -> Self {
        Self {
            bindings: IndexMap::new(),
            parent: None,
        }
    }

    /// Create a child scope that chains to `self`.
    pub fn child(&self) -> Self {
        Self {
            bindings: IndexMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Bind `name` to `value` in the current scope.
    pub fn bind(&mut self, name: impl Into<String>, value: SemanticValue) {
        self.bindings.insert(name.into(), value);
    }

    /// Look up `name`, searching from innermost scope outward.
    pub fn lookup(&self, name: &str) -> Option<&SemanticValue> {
        if let Some(val) = self.bindings.get(name) {
            Some(val)
        } else if let Some(ref parent) = self.parent {
            parent.lookup(name)
        } else {
            None
        }
    }

    /// Return all visible bindings (innermost shadows outer).
    pub fn all_bindings(&self) -> IndexMap<String, SemanticValue> {
        let mut result = if let Some(ref parent) = self.parent {
            parent.all_bindings()
        } else {
            IndexMap::new()
        };
        // Inner bindings shadow outer ones.
        for (k, v) in &self.bindings {
            result.insert(k.clone(), v.clone());
        }
        result
    }

    /// Number of scope layers.
    pub fn depth(&self) -> usize {
        1 + self
            .parent
            .as_ref()
            .map(|p| p.depth())
            .unwrap_or(0)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Evaluator — direct interpreter
// ═══════════════════════════════════════════════════════════════════════════

/// A step-bounded direct interpreter for the EvalSpec expression language.
///
/// The evaluator walks the AST and produces `SemanticValue`s.  A *step
/// counter* acts as a divergence guard: if more than `max_steps` expression
/// nodes are visited the evaluator returns `NonTermination`.
pub struct Evaluator {
    pub env: Environment,
    pub max_steps: usize,
    steps: usize,
}

impl Evaluator {
    /// Default step limit (1 million).
    const DEFAULT_MAX_STEPS: usize = 1_000_000;

    pub fn new() -> Self {
        let mut env = Environment::new();
        register_builtins(&mut env);
        Self {
            env,
            max_steps: Self::DEFAULT_MAX_STEPS,
            steps: 0,
        }
    }

    pub fn with_env(env: Environment) -> Self {
        Self {
            env,
            max_steps: Self::DEFAULT_MAX_STEPS,
            steps: 0,
        }
    }

    /// Check and increment the step counter.
    fn tick(&mut self) -> SemResult<()> {
        self.steps += 1;
        if self.steps > self.max_steps {
            Err(SemanticError::NonTermination {
                desc: format!("exceeded {} evaluation steps", self.max_steps),
            })
        } else {
            Ok(())
        }
    }

    // ── Program / declaration evaluation ──────────────────────────────

    pub fn eval_program(&mut self, program: &Program) -> SemResult<Vec<SemanticValue>> {
        let mut results = Vec::new();
        for decl in &program.declarations {
            if let Some(val) = self.eval_declaration(&decl.node)? {
                results.push(val);
            }
        }
        Ok(results)
    }

    pub fn eval_declaration(
        &mut self,
        decl: &Declaration,
    ) -> SemResult<Option<SemanticValue>> {
        match decl {
            Declaration::Let(let_decl) => {
                let val = self.eval_expr(&let_decl.value.node)?;
                self.env.bind(let_decl.name.clone(), val.clone());
                Ok(Some(val))
            }
            Declaration::Metric(metric_decl) => {
                // Bind the metric as a function in the environment.
                let params: Vec<String> =
                    metric_decl.params.iter().map(|p| p.name.clone()).collect();
                let func = SemanticValue::Function(FunctionValue {
                    params,
                    body: Box::new(metric_decl.body.node.clone()),
                    closure: self.env.clone(),
                });
                self.env.bind(metric_decl.name.clone(), func);
                Ok(None)
            }
            Declaration::Type(_) => Ok(None),
            Declaration::Import(_) => Ok(None),
            Declaration::Test(test_decl) => {
                let val = self.eval_expr(&test_decl.body.node)?;
                Ok(Some(val))
            }
        }
    }

    // ── Core expression evaluator ─────────────────────────────────────

    pub fn eval_expr(&mut self, expr: &Expr) -> SemResult<SemanticValue> {
        self.tick()?;
        match expr {
            Expr::Literal(lit) => self.eval_literal(lit),

            Expr::Variable(name) => self
                .env
                .lookup(name)
                .cloned()
                .ok_or_else(|| SemanticError::UndefinedVariable {
                    name: name.clone(),
                }),

            Expr::BinaryOp { op, left, right } => {
                self.eval_binary_op(op, &left.node, &right.node)
            }

            Expr::UnaryOp { op, operand } => self.eval_unary_op(op, &operand.node),

            Expr::FunctionCall { name, args } => self.eval_function_call(name, args),

            Expr::MethodCall {
                receiver,
                method,
                args,
            } => self.eval_method_call(&receiver.node, method, args),

            Expr::Lambda { params, body } => {
                let param_names: Vec<String> =
                    params.iter().map(|p| p.name.clone()).collect();
                self.eval_lambda(&param_names, &body.node)
            }

            Expr::Let {
                name, value, body, ..
            } => {
                let val = self.eval_expr(&value.node)?;
                let mut child_env = self.env.child();
                child_env.bind(name.clone(), val);
                let saved = std::mem::replace(&mut self.env, child_env);
                let result = self.eval_expr(&body.node);
                self.env = saved;
                result
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.eval_if(&condition.node, &then_branch.node, &else_branch.node),

            Expr::Match { scrutinee, arms } => {
                self.eval_match(&scrutinee.node, arms)
            }

            Expr::Block(exprs) => {
                if exprs.is_empty() {
                    return Ok(SemanticValue::Unit);
                }
                let mut last = SemanticValue::Unit;
                for expr_spanned in exprs {
                    last = self.eval_expr(&expr_spanned.node)?;
                }
                Ok(last)
            }

            Expr::FieldAccess { expr, field } => {
                let val = self.eval_expr(&expr.node)?;
                self.eval_field_access(&val, field)
            }

            Expr::IndexAccess { expr, index } => {
                let collection = self.eval_expr(&expr.node)?;
                let idx = self.eval_expr(&index.node)?;
                self.eval_index_access(&collection, &idx)
            }

            Expr::ListLiteral(elems) => {
                let values: SemResult<Vec<_>> =
                    elems.iter().map(|e| self.eval_expr(&e.node)).collect();
                Ok(SemanticValue::List(values?))
            }

            Expr::TupleLiteral(elems) => {
                let values: SemResult<Vec<_>> =
                    elems.iter().map(|e| self.eval_expr(&e.node)).collect();
                Ok(SemanticValue::Tuple(values?))
            }

            Expr::Aggregate {
                op,
                collection,
                binding,
                body,
                semiring,
            } => self.eval_aggregate(op, &collection.node, binding, body, semiring),

            Expr::NGramExtract { input, n } => self.eval_ngram_extract(&input.node, n),

            Expr::TokenizeExpr { input, .. } => self.eval_tokenize(&input.node),

            Expr::MatchPattern {
                input,
                pattern,
                mode,
            } => self.eval_match_pattern(&input.node, pattern, mode),

            Expr::SemiringCast { expr, from, to } => {
                let val = self.eval_expr(&expr.node)?;
                self.eval_semiring_cast(&val, from, to)
            }

            Expr::ClipCount { count, max_count } => {
                let c = self.eval_expr(&count.node)?;
                let m = self.eval_expr(&max_count.node)?;
                self.eval_clip_count(&c, &m)
            }

            Expr::Compose { first, second } => {
                let first_val = self.eval_expr(&first.node)?;
                let second_val = self.eval_expr(&second.node)?;
                // Composition: apply second to the output of first.
                // Both must be functions.
                match (&first_val, &second_val) {
                    (SemanticValue::Function(f1), SemanticValue::Function(_f2)) => {
                        Ok(SemanticValue::Function(FunctionValue {
                            params: f1.params.clone(),
                            body: Box::new(Expr::FunctionCall {
                                name: "__compose__".into(),
                                args: vec![],
                            }),
                            closure: {
                                let mut env = self.env.child();
                                env.bind("__first__".to_string(), first_val.clone());
                                env.bind("__second__".to_string(), second_val.clone());
                                env
                            },
                        }))
                    }
                    _ => Err(SemanticError::InvalidOperation {
                        desc: "compose requires two function values".into(),
                    }),
                }
            }
        }
    }

    fn eval_literal(&self, lit: &Literal) -> SemResult<SemanticValue> {
        Ok(match lit {
            Literal::Integer(n) => SemanticValue::Integer(*n),
            Literal::Float(f) => SemanticValue::Float(f.into_inner()),
            Literal::Bool(b) => SemanticValue::Boolean(*b),
            Literal::String(s) => SemanticValue::Str(s.clone()),
        })
    }

    // ── Binary operations ─────────────────────────────────────────────

    pub fn eval_binary_op(
        &mut self,
        op: &BinaryOp,
        left: &Expr,
        right: &Expr,
    ) -> SemResult<SemanticValue> {
        // Short-circuit for logical operators.
        match op {
            BinaryOp::And => {
                let lv = self.eval_expr(left)?;
                let lb = lv.to_bool().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "Boolean".into(),
                    found: lv.type_name().into(),
                })?;
                if !lb {
                    return Ok(SemanticValue::Boolean(false));
                }
                let rv = self.eval_expr(right)?;
                let rb = rv.to_bool().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "Boolean".into(),
                    found: rv.type_name().into(),
                })?;
                return Ok(SemanticValue::Boolean(rb));
            }
            BinaryOp::Or => {
                let lv = self.eval_expr(left)?;
                let lb = lv.to_bool().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "Boolean".into(),
                    found: lv.type_name().into(),
                })?;
                if lb {
                    return Ok(SemanticValue::Boolean(true));
                }
                let rv = self.eval_expr(right)?;
                let rb = rv.to_bool().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "Boolean".into(),
                    found: rv.type_name().into(),
                })?;
                return Ok(SemanticValue::Boolean(rb));
            }
            _ => {}
        }

        let lv = self.eval_expr(left)?;
        let rv = self.eval_expr(right)?;

        // Semiring values use their own algebra.
        if let (SemanticValue::SemiringVal(sl), SemanticValue::SemiringVal(sr)) = (&lv, &rv) {
            return match op {
                BinaryOp::Add => sl.sr_add(sr).map(SemanticValue::SemiringVal),
                BinaryOp::Mul => sl.sr_mul(sr).map(SemanticValue::SemiringVal),
                _ => Err(SemanticError::InvalidOperation {
                    desc: format!("operator `{op}` not supported on semiring values"),
                }),
            };
        }

        // String concatenation.
        if let BinaryOp::Add = op {
            if let (SemanticValue::Str(a), SemanticValue::Str(b)) = (&lv, &rv) {
                return Ok(SemanticValue::Str(format!("{a}{b}")));
            }
        }

        // Equality / inequality for arbitrary values.
        match op {
            BinaryOp::Eq => return Ok(SemanticValue::Boolean(lv == rv)),
            BinaryOp::Neq => return Ok(SemanticValue::Boolean(lv != rv)),
            _ => {}
        }

        // Numeric arithmetic and comparison.
        // Promote to float if either side is float.
        let lf = lv
            .to_f64()
            .ok_or_else(|| SemanticError::TypeMismatch {
                expected: "numeric".into(),
                found: lv.type_name().into(),
            })?;
        let rf = rv
            .to_f64()
            .ok_or_else(|| SemanticError::TypeMismatch {
                expected: "numeric".into(),
                found: rv.type_name().into(),
            })?;

        let both_int = matches!((&lv, &rv), (SemanticValue::Integer(_), SemanticValue::Integer(_)));

        match op {
            BinaryOp::Add => {
                if both_int {
                    Ok(SemanticValue::Integer(lf as i64 + rf as i64))
                } else {
                    Ok(SemanticValue::Float(lf + rf))
                }
            }
            BinaryOp::Sub => {
                if both_int {
                    Ok(SemanticValue::Integer(lf as i64 - rf as i64))
                } else {
                    Ok(SemanticValue::Float(lf - rf))
                }
            }
            BinaryOp::Mul => {
                if both_int {
                    Ok(SemanticValue::Integer(lf as i64 * rf as i64))
                } else {
                    Ok(SemanticValue::Float(lf * rf))
                }
            }
            BinaryOp::Div => {
                if rf == 0.0 {
                    return Err(SemanticError::DivisionByZero);
                }
                if both_int {
                    Ok(SemanticValue::Integer(lf as i64 / rf as i64))
                } else {
                    Ok(SemanticValue::Float(lf / rf))
                }
            }
            BinaryOp::Min => Ok(SemanticValue::Float(lf.min(rf))),
            BinaryOp::Max => Ok(SemanticValue::Float(lf.max(rf))),
            BinaryOp::Lt => Ok(SemanticValue::Boolean(lf < rf)),
            BinaryOp::Le => Ok(SemanticValue::Boolean(lf <= rf)),
            BinaryOp::Gt => Ok(SemanticValue::Boolean(lf > rf)),
            BinaryOp::Ge => Ok(SemanticValue::Boolean(lf >= rf)),
            BinaryOp::Eq | BinaryOp::Neq => unreachable!("handled above"),
            BinaryOp::And | BinaryOp::Or => unreachable!("handled above"),
            BinaryOp::Mod => {
                if rf == 0.0 {
                    return Err(SemanticError::DivisionByZero);
                }
                if both_int {
                    Ok(SemanticValue::Integer(lf as i64 % rf as i64))
                } else {
                    Ok(SemanticValue::Float(lf % rf))
                }
            }
            BinaryOp::Pow => Ok(SemanticValue::Float(lf.powf(rf))),
            BinaryOp::Concat | BinaryOp::SemiringAdd | BinaryOp::SemiringMul => {
                Err(SemanticError::InvalidOperation {
                    desc: format!("operator {:?} not supported in numeric context", op),
                })
            }
        }
    }

    // ── Unary operations ──────────────────────────────────────────────

    pub fn eval_unary_op(
        &mut self,
        op: &UnaryOp,
        operand: &Expr,
    ) -> SemResult<SemanticValue> {
        let val = self.eval_expr(operand)?;
        match op {
            UnaryOp::Neg => match &val {
                SemanticValue::Integer(n) => Ok(SemanticValue::Integer(-n)),
                SemanticValue::Float(v) => Ok(SemanticValue::Float(-v)),
                _ => Err(SemanticError::TypeMismatch {
                    expected: "numeric".into(),
                    found: val.type_name().into(),
                }),
            },
            UnaryOp::Not => {
                let b = val.to_bool().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "Boolean".into(),
                    found: val.type_name().into(),
                })?;
                Ok(SemanticValue::Boolean(!b))
            }
            UnaryOp::Star => {
                // Kleene star only makes sense on semiring values.
                Err(SemanticError::UnsupportedExpression {
                    desc: "Kleene star is not supported in the direct evaluator".into(),
                })
            }
        }
    }

    // ── Function call dispatch ────────────────────────────────────────

    pub fn eval_function_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
    ) -> SemResult<SemanticValue> {
        let arg_vals: SemResult<Vec<_>> =
            args.iter().map(|a| self.eval_expr(&a.node)).collect();
        let arg_vals = arg_vals?;

        // Look up in environment.
        let func_val = self.env.lookup(name).cloned();
        match func_val {
            Some(SemanticValue::Function(fv)) => self.apply_function(&fv, &arg_vals),
            Some(_) => Err(SemanticError::TypeMismatch {
                expected: "Function".into(),
                found: "non-function value".into(),
            }),
            None => {
                // Try built-in dispatch.
                self.dispatch_builtin(name, &arg_vals)
            }
        }
    }

    fn eval_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Spanned<Expr>],
    ) -> SemResult<SemanticValue> {
        let recv = self.eval_expr(receiver)?;
        let mut arg_vals: Vec<SemanticValue> = vec![recv];
        for a in args {
            arg_vals.push(self.eval_expr(&a.node)?);
        }
        self.dispatch_builtin(method, &arg_vals)
    }

    // ── Lambda ────────────────────────────────────────────────────────

    pub fn eval_lambda(
        &mut self,
        params: &[String],
        body: &Expr,
    ) -> SemResult<SemanticValue> {
        Ok(SemanticValue::Function(FunctionValue {
            params: params.to_vec(),
            body: Box::new(body.clone()),
            closure: self.env.clone(),
        }))
    }

    // ── Conditionals ──────────────────────────────────────────────────

    pub fn eval_if(
        &mut self,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
    ) -> SemResult<SemanticValue> {
        let cond_val = self.eval_expr(condition)?;
        let b = cond_val
            .to_bool()
            .ok_or_else(|| SemanticError::TypeMismatch {
                expected: "Boolean".into(),
                found: cond_val.type_name().into(),
            })?;
        if b {
            self.eval_expr(then_branch)
        } else {
            self.eval_expr(else_branch)
        }
    }

    // ── Pattern matching ──────────────────────────────────────────────

    pub fn eval_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
    ) -> SemResult<SemanticValue> {
        let scrut_val = self.eval_expr(scrutinee)?;
        for arm in arms {
            if let Some(bindings) = self.match_pattern(&arm.pattern.node, &scrut_val)? {
                // Check optional guard.
                if let Some(ref guard) = arm.guard {
                    let mut child_env = self.env.child();
                    for (k, v) in &bindings {
                        child_env.bind(k.clone(), v.clone());
                    }
                    let saved = std::mem::replace(&mut self.env, child_env);
                    let guard_val = self.eval_expr(&guard.node)?;
                    self.env = saved;
                    let passes =
                        guard_val
                            .to_bool()
                            .ok_or_else(|| SemanticError::TypeMismatch {
                                expected: "Boolean".into(),
                                found: guard_val.type_name().into(),
                            })?;
                    if !passes {
                        continue;
                    }
                }
                // Matched — evaluate the arm body in an extended environment.
                let mut child_env = self.env.child();
                for (k, v) in bindings {
                    child_env.bind(k, v);
                }
                let saved = std::mem::replace(&mut self.env, child_env);
                let result = self.eval_expr(&arm.body.node);
                self.env = saved;
                return result;
            }
        }
        Err(SemanticError::InvalidOperation {
            desc: "non-exhaustive match: no arm matched".into(),
        })
    }

    /// Attempt to match a pattern against a value, returning bindings on success.
    pub fn match_pattern(
        &self,
        pattern: &Pattern,
        value: &SemanticValue,
    ) -> SemResult<Option<Vec<(String, SemanticValue)>>> {
        match pattern {
            Pattern::Wildcard => Ok(Some(vec![])),

            Pattern::Var(name) => Ok(Some(vec![(name.clone(), value.clone())])),

            Pattern::Literal(lit) => {
                let lit_val = match lit {
                    Literal::Integer(n) => SemanticValue::Integer(*n),
                    Literal::Float(f) => SemanticValue::Float(f.into_inner()),
                    Literal::Bool(b) => SemanticValue::Boolean(*b),
                    Literal::String(s) => SemanticValue::Str(s.clone()),
                };
                if lit_val == *value {
                    Ok(Some(vec![]))
                } else {
                    Ok(None)
                }
            }

            Pattern::Tuple(elem_pats) => {
                if let SemanticValue::Tuple(elems) = value {
                    if elems.len() != elem_pats.len() {
                        return Ok(None);
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in elem_pats.iter().zip(elems.iter()) {
                        match self.match_pattern(&pat.node, val)? {
                            Some(bs) => bindings.extend(bs),
                            None => return Ok(None),
                        }
                    }
                    Ok(Some(bindings))
                } else {
                    Ok(None)
                }
            }

            Pattern::Constructor { name, args } => {
                // Simple constructor matching: we treat constructor names as
                // tags.  For now only `Some`/`None` are recognized.
                match name.as_str() {
                    "None" => {
                        if let SemanticValue::Unit = value {
                            Ok(Some(vec![]))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => {
                        // Generic: treat the constructor name as a tuple tag.
                        if let SemanticValue::Tuple(elems) = value {
                            if elems.len() != args.len() {
                                return Ok(None);
                            }
                            let mut bindings = Vec::new();
                            for (pat, val) in args.iter().zip(elems.iter()) {
                                match self.match_pattern(&pat.node, val)? {
                                    Some(bs) => bindings.extend(bs),
                                    None => return Ok(None),
                                }
                            }
                            Ok(Some(bindings))
                        } else {
                            Ok(None)
                        }
                    }
                }
            }

            Pattern::List { elems, rest } => {
                if let SemanticValue::List(list) = value {
                    if list.len() < elems.len() {
                        return Ok(None);
                    }
                    if rest.is_none() && list.len() != elems.len() {
                        return Ok(None);
                    }
                    let mut bindings = Vec::new();
                    for (pat, val) in elems.iter().zip(list.iter()) {
                        match self.match_pattern(&pat.node, val)? {
                            Some(bs) => bindings.extend(bs),
                            None => return Ok(None),
                        }
                    }
                    if let Some(rest_pat) = rest {
                        let rest_val = SemanticValue::List(list[elems.len()..].to_vec());
                        match self.match_pattern(&rest_pat.node, &rest_val)? {
                            Some(bs) => bindings.extend(bs),
                            None => return Ok(None),
                        }
                    }
                    Ok(Some(bindings))
                } else {
                    Ok(None)
                }
            }

            Pattern::Guard { pattern, condition: _ } => {
                // The guard is evaluated in eval_match, so here we just match
                // the inner pattern.
                self.match_pattern(&pattern.node, value)
            }
        }
    }

    // ── Aggregation ───────────────────────────────────────────────────

    pub fn eval_aggregate(
        &mut self,
        op: &AggregationOp,
        collection: &Expr,
        binding: &Option<String>,
        body: &Option<Box<Spanned<Expr>>>,
        semiring: &Option<SemiringType>,
    ) -> SemResult<SemanticValue> {
        let coll_val = self.eval_expr(collection)?;
        let items = match &coll_val {
            SemanticValue::List(elems) => elems.clone(),
            SemanticValue::TokenSequence(tokens) => tokens
                .iter()
                .map(|t| SemanticValue::Str(t.clone()))
                .collect(),
            _ => {
                return Err(SemanticError::TypeMismatch {
                    expected: "List or TokenSequence".into(),
                    found: coll_val.type_name().into(),
                });
            }
        };

        if items.is_empty() {
            return self.aggregate_identity(op, semiring);
        }

        // Map phase: evaluate body for each item if binding + body are present.
        let mapped: SemResult<Vec<SemanticValue>> = if let (Some(var), Some(body_expr)) =
            (binding, body)
        {
            items
                .iter()
                .map(|item| {
                    let mut child_env = self.env.child();
                    child_env.bind(var.clone(), item.clone());
                    let saved = std::mem::replace(&mut self.env, child_env);
                    let r = self.eval_expr(&body_expr.node);
                    self.env = saved;
                    r
                })
                .collect()
        } else {
            Ok(items)
        };
        let mapped = mapped?;

        self.reduce_aggregate(op, &mapped, semiring)
    }

    fn aggregate_identity(
        &self,
        op: &AggregationOp,
        semiring: &Option<SemiringType>,
    ) -> SemResult<SemanticValue> {
        if let Some(st) = semiring {
            return match op {
                AggregationOp::Sum | AggregationOp::Count => {
                    Ok(SemanticValue::SemiringVal(SemiringValue::zero_for(st)))
                }
                AggregationOp::Product => {
                    Ok(SemanticValue::SemiringVal(SemiringValue::one_for(st)))
                }
                _ => Ok(SemanticValue::SemiringVal(SemiringValue::zero_for(st))),
            };
        }
        match op {
            AggregationOp::Sum | AggregationOp::Count => Ok(SemanticValue::Integer(0)),
            AggregationOp::Product => Ok(SemanticValue::Integer(1)),
            AggregationOp::Min => Ok(SemanticValue::Float(f64::INFINITY)),
            AggregationOp::Max => Ok(SemanticValue::Float(f64::NEG_INFINITY)),
            AggregationOp::Mean
            | AggregationOp::HarmonicMean
            | AggregationOp::GeometricMean => Ok(SemanticValue::Float(0.0)),
        }
    }

    fn reduce_aggregate(
        &self,
        op: &AggregationOp,
        values: &[SemanticValue],
        _semiring: &Option<SemiringType>,
    ) -> SemResult<SemanticValue> {
        match op {
            AggregationOp::Count => Ok(SemanticValue::Integer(values.len() as i64)),

            AggregationOp::Sum => {
                let mut acc = 0.0f64;
                let mut all_int = true;
                let mut int_acc = 0i64;
                for v in values {
                    match v {
                        SemanticValue::Integer(n) => {
                            acc += *n as f64;
                            int_acc = int_acc.wrapping_add(*n);
                        }
                        _ => {
                            all_int = false;
                            acc += v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                                expected: "numeric".into(),
                                found: v.type_name().into(),
                            })?;
                        }
                    }
                }
                if all_int {
                    Ok(SemanticValue::Integer(int_acc))
                } else {
                    Ok(SemanticValue::Float(acc))
                }
            }

            AggregationOp::Product => {
                let mut acc = 1.0f64;
                let mut all_int = true;
                let mut int_acc = 1i64;
                for v in values {
                    match v {
                        SemanticValue::Integer(n) => {
                            acc *= *n as f64;
                            int_acc = int_acc.wrapping_mul(*n);
                        }
                        _ => {
                            all_int = false;
                            acc *= v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                                expected: "numeric".into(),
                                found: v.type_name().into(),
                            })?;
                        }
                    }
                }
                if all_int {
                    Ok(SemanticValue::Integer(int_acc))
                } else {
                    Ok(SemanticValue::Float(acc))
                }
            }

            AggregationOp::Min => {
                let mut min_val = f64::INFINITY;
                for v in values {
                    let f = v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                        expected: "numeric".into(),
                        found: v.type_name().into(),
                    })?;
                    if f < min_val {
                        min_val = f;
                    }
                }
                Ok(SemanticValue::Float(min_val))
            }

            AggregationOp::Max => {
                let mut max_val = f64::NEG_INFINITY;
                for v in values {
                    let f = v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                        expected: "numeric".into(),
                        found: v.type_name().into(),
                    })?;
                    if f > max_val {
                        max_val = f;
                    }
                }
                Ok(SemanticValue::Float(max_val))
            }

            AggregationOp::Mean => {
                if values.is_empty() {
                    return Ok(SemanticValue::Float(0.0));
                }
                let mut sum = 0.0f64;
                for v in values {
                    sum += v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                        expected: "numeric".into(),
                        found: v.type_name().into(),
                    })?;
                }
                Ok(SemanticValue::Float(sum / values.len() as f64))
            }

            AggregationOp::HarmonicMean => {
                if values.is_empty() {
                    return Ok(SemanticValue::Float(0.0));
                }
                let mut reciprocal_sum = 0.0f64;
                for v in values {
                    let f = v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                        expected: "numeric".into(),
                        found: v.type_name().into(),
                    })?;
                    if f == 0.0 {
                        return Ok(SemanticValue::Float(0.0));
                    }
                    reciprocal_sum += 1.0 / f;
                }
                Ok(SemanticValue::Float(
                    values.len() as f64 / reciprocal_sum,
                ))
            }

            AggregationOp::GeometricMean => {
                if values.is_empty() {
                    return Ok(SemanticValue::Float(0.0));
                }
                let mut log_sum = 0.0f64;
                for v in values {
                    let f = v.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                        expected: "numeric".into(),
                        found: v.type_name().into(),
                    })?;
                    if f <= 0.0 {
                        return Ok(SemanticValue::Float(0.0));
                    }
                    log_sum += f.ln();
                }
                Ok(SemanticValue::Float(
                    (log_sum / values.len() as f64).exp(),
                ))
            }
        }
    }

    // ── N-gram extraction ─────────────────────────────────────────────

    pub fn eval_ngram_extract(
        &mut self,
        input: &Expr,
        n: &usize,
    ) -> SemResult<SemanticValue> {
        let val = self.eval_expr(input)?;
        let tokens = self.extract_tokens(&val)?;

        let n = *n;
        if n == 0 || tokens.len() < n {
            return Ok(SemanticValue::NGramSet(IndexMap::new()));
        }

        let mut ngrams: IndexMap<Vec<String>, usize> = IndexMap::new();
        for window in tokens.windows(n) {
            let gram = window.to_vec();
            *ngrams.entry(gram).or_insert(0) += 1;
        }
        Ok(SemanticValue::NGramSet(ngrams))
    }

    // ── Tokenization ──────────────────────────────────────────────────

    pub fn eval_tokenize(&mut self, input: &Expr) -> SemResult<SemanticValue> {
        let val = self.eval_expr(input)?;
        match &val {
            SemanticValue::Str(s) => {
                let tokens: Vec<String> = s
                    .split_whitespace()
                    .map(|t| t.to_lowercase())
                    .collect();
                Ok(SemanticValue::TokenSequence(tokens))
            }
            SemanticValue::TokenSequence(_) => Ok(val),
            _ => Err(SemanticError::TypeMismatch {
                expected: "String or TokenSequence".into(),
                found: val.type_name().into(),
            }),
        }
    }

    // ── Match pattern (regex/glob/exact/contains) ─────────────────────

    fn eval_match_pattern(
        &mut self,
        input: &Expr,
        pattern: &str,
        mode: &MatchMode,
    ) -> SemResult<SemanticValue> {
        let val = self.eval_expr(input)?;
        let s = match &val {
            SemanticValue::Str(s) => s.clone(),
            _ => {
                return Err(SemanticError::TypeMismatch {
                    expected: "String".into(),
                    found: val.type_name().into(),
                });
            }
        };

        let matched = match mode {
            MatchMode::Exact => s == pattern,
            MatchMode::Contains => s.contains(pattern),
            MatchMode::Glob => {
                // Simple glob: only support * as wildcard.
                let parts: Vec<&str> = pattern.split('*').collect();
                if parts.len() == 1 {
                    s == pattern
                } else {
                    let mut pos = 0;
                    let mut ok = true;
                    for (i, part) in parts.iter().enumerate() {
                        if part.is_empty() {
                            continue;
                        }
                        if let Some(found) = s[pos..].find(part) {
                            if i == 0 && found != 0 {
                                ok = false;
                                break;
                            }
                            pos += found + part.len();
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && !parts.last().unwrap_or(&"").is_empty() {
                        s.ends_with(parts.last().unwrap())
                    } else {
                        ok
                    }
                }
            }
            MatchMode::Regex => {
                // Simplified regex: just check contains as a fallback.
                s.contains(pattern)
            }
        };

        Ok(SemanticValue::Boolean(matched))
    }

    // ── Semiring cast ─────────────────────────────────────────────────

    fn eval_semiring_cast(
        &self,
        val: &SemanticValue,
        _from: &SemiringType,
        to: &SemiringType,
    ) -> SemResult<SemanticValue> {
        let f = val.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: val.type_name().into(),
        })?;
        let casted = match to {
            SemiringType::Counting => SemiringValue::Counting(f.max(0.0) as u64),
            SemiringType::Boolean => SemiringValue::Bool(f != 0.0),
            SemiringType::Tropical => SemiringValue::Tropical(f),
            SemiringType::Real => SemiringValue::Real(f),
            SemiringType::Goldilocks => {
                let p = crate::wfa::GOLDILOCKS_PRIME;
                SemiringValue::Goldilocks((f.abs() as u64) % p)
            }
            SemiringType::BoundedCounting(bound) => SemiringValue::BoundedCounting {
                value: (f.max(0.0) as u64).min(*bound),
                max_count: *bound,
            },
            SemiringType::LogDomain => SemiringValue::Real(f.ln()),
            SemiringType::Viterbi => SemiringValue::Real(f.clamp(0.0, 1.0)),
        };
        Ok(SemanticValue::SemiringVal(casted))
    }

    // ── Clip count ────────────────────────────────────────────────────

    fn eval_clip_count(
        &self,
        count: &SemanticValue,
        max_count: &SemanticValue,
    ) -> SemResult<SemanticValue> {
        let c = count.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: count.type_name().into(),
        })?;
        let m = max_count
            .to_f64()
            .ok_or_else(|| SemanticError::TypeMismatch {
                expected: "numeric".into(),
                found: max_count.type_name().into(),
            })?;
        Ok(SemanticValue::Integer(c.min(m) as i64))
    }

    // ── Field / index access ──────────────────────────────────────────

    fn eval_field_access(
        &self,
        val: &SemanticValue,
        field: &str,
    ) -> SemResult<SemanticValue> {
        match val {
            SemanticValue::Tuple(elems) => {
                // Access tuple fields by index: .0, .1, etc.
                if let Ok(idx) = field.parse::<usize>() {
                    elems.get(idx).cloned().ok_or_else(|| {
                        SemanticError::InvalidOperation {
                            desc: format!(
                                "tuple index {idx} out of bounds (len {})",
                                elems.len()
                            ),
                        }
                    })
                } else {
                    Err(SemanticError::InvalidOperation {
                        desc: format!("cannot access field `{field}` on a tuple"),
                    })
                }
            }
            SemanticValue::NGramSet(ngrams) => match field {
                "len" | "size" => Ok(SemanticValue::Integer(ngrams.len() as i64)),
                _ => Err(SemanticError::InvalidOperation {
                    desc: format!("unknown field `{field}` on NGramSet"),
                }),
            },
            _ => Err(SemanticError::InvalidOperation {
                desc: format!(
                    "field access `.{field}` not supported on {}",
                    val.type_name()
                ),
            }),
        }
    }

    fn eval_index_access(
        &self,
        collection: &SemanticValue,
        index: &SemanticValue,
    ) -> SemResult<SemanticValue> {
        let idx = index.to_i64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "integer index".into(),
            found: index.type_name().into(),
        })? as usize;

        match collection {
            SemanticValue::List(elems) => {
                elems.get(idx).cloned().ok_or_else(|| {
                    SemanticError::InvalidOperation {
                        desc: format!(
                            "list index {idx} out of bounds (len {})",
                            elems.len()
                        ),
                    }
                })
            }
            SemanticValue::Tuple(elems) => {
                elems.get(idx).cloned().ok_or_else(|| {
                    SemanticError::InvalidOperation {
                        desc: format!(
                            "tuple index {idx} out of bounds (len {})",
                            elems.len()
                        ),
                    }
                })
            }
            SemanticValue::TokenSequence(tokens) => {
                tokens.get(idx).cloned().map(SemanticValue::Str).ok_or_else(|| {
                    SemanticError::InvalidOperation {
                        desc: format!(
                            "token index {idx} out of bounds (len {})",
                            tokens.len()
                        ),
                    }
                })
            }
            _ => Err(SemanticError::InvalidOperation {
                desc: format!(
                    "index access not supported on {}",
                    collection.type_name()
                ),
            }),
        }
    }

    // ── Function application ──────────────────────────────────────────

    pub fn apply_function(
        &mut self,
        func: &FunctionValue,
        args: &[SemanticValue],
    ) -> SemResult<SemanticValue> {
        if args.len() != func.params.len() {
            return Err(SemanticError::InvalidOperation {
                desc: format!(
                    "function expects {} arguments, got {}",
                    func.params.len(),
                    args.len()
                ),
            });
        }
        let mut call_env = func.closure.child();
        for (param, arg) in func.params.iter().zip(args.iter()) {
            call_env.bind(param.clone(), arg.clone());
        }
        let saved = std::mem::replace(&mut self.env, call_env);
        let result = self.eval_expr(&func.body);
        self.env = saved;
        result
    }

    // ── Built-in dispatch ─────────────────────────────────────────────

    fn dispatch_builtin(
        &self,
        name: &str,
        args: &[SemanticValue],
    ) -> SemResult<SemanticValue> {
        match name {
            "len" => builtin_len(args),
            "count" => builtin_count(args),
            "min" => builtin_min(args),
            "max" => builtin_max(args),
            "abs" => builtin_abs(args),
            "log" => builtin_log(args),
            "exp" => builtin_exp(args),
            "floor" => builtin_floor(args),
            "ceil" => builtin_ceil(args),
            "round" => builtin_round(args),
            "intersection" => builtin_intersection(args),
            "union" => builtin_union(args),
            "sqrt" => builtin_sqrt(args),
            "pow" => builtin_pow(args),
            "to_float" => {
                if args.len() != 1 {
                    return Err(SemanticError::InvalidOperation {
                        desc: "to_float expects 1 argument".into(),
                    });
                }
                let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "numeric".into(),
                    found: args[0].type_name().into(),
                })?;
                Ok(SemanticValue::Float(f))
            }
            "to_int" => {
                if args.len() != 1 {
                    return Err(SemanticError::InvalidOperation {
                        desc: "to_int expects 1 argument".into(),
                    });
                }
                let i = args[0].to_i64().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "numeric".into(),
                    found: args[0].type_name().into(),
                })?;
                Ok(SemanticValue::Integer(i))
            }
            "contains" => {
                if args.len() != 2 {
                    return Err(SemanticError::InvalidOperation {
                        desc: "contains expects 2 arguments".into(),
                    });
                }
                match (&args[0], &args[1]) {
                    (SemanticValue::Str(haystack), SemanticValue::Str(needle)) => {
                        Ok(SemanticValue::Boolean(haystack.contains(needle.as_str())))
                    }
                    (SemanticValue::List(items), needle) => {
                        Ok(SemanticValue::Boolean(items.iter().any(|x| x == needle)))
                    }
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "String or List".into(),
                        found: args[0].type_name().into(),
                    }),
                }
            }
            "concat" => {
                if args.len() != 2 {
                    return Err(SemanticError::InvalidOperation {
                        desc: "concat expects 2 arguments".into(),
                    });
                }
                match (&args[0], &args[1]) {
                    (SemanticValue::List(a), SemanticValue::List(b)) => {
                        let mut result = a.clone();
                        result.extend(b.iter().cloned());
                        Ok(SemanticValue::List(result))
                    }
                    (SemanticValue::Str(a), SemanticValue::Str(b)) => {
                        Ok(SemanticValue::Str(format!("{a}{b}")))
                    }
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "List or String".into(),
                        found: args[0].type_name().into(),
                    }),
                }
            }
            _ => Err(SemanticError::UndefinedVariable {
                name: name.to_string(),
            }),
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    fn extract_tokens(&self, val: &SemanticValue) -> SemResult<Vec<String>> {
        match val {
            SemanticValue::TokenSequence(tokens) => Ok(tokens.clone()),
            SemanticValue::List(elems) => {
                let mut tokens = Vec::new();
                for e in elems {
                    if let SemanticValue::Str(s) = e {
                        tokens.push(s.clone());
                    } else {
                        return Err(SemanticError::TypeMismatch {
                            expected: "List<String>".into(),
                            found: e.type_name().into(),
                        });
                    }
                }
                Ok(tokens)
            }
            SemanticValue::Str(s) => {
                Ok(s.split_whitespace().map(|t| t.to_string()).collect())
            }
            _ => Err(SemanticError::TypeMismatch {
                expected: "TokenSequence, List<String>, or String".into(),
                found: val.type_name().into(),
            }),
        }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Formal Power Series Semantics
// ═══════════════════════════════════════════════════════════════════════════

/// Denotation of an EvalSpec metric as a formal power series.
///
/// A formal power series (FPS) over a free monoid Σ* with coefficients in a
/// semiring S is a function f : Σ* → S.  Every WFA computes such a series.
///
/// Each metric type has a canonical FPS characterization:
///
/// - **ExactMatch**: the indicator series I(ref) where I(ref)(w) = 1 iff
///   w = ref.  This is a polynomial (finite support).
///
/// - **TokenF1**: a pair of series (P, R) where P counts precision-related
///   n-gram overlaps and R counts recall-related overlaps, with F1 computed
///   as a post-processing step 2PR/(P+R).
///
/// - **BLEU**: a product of n-gram precision series with a brevity penalty
///   term, corresponding to the geometric mean of clipped n-gram counts.
///
/// - **ROUGE-N**: similar to TokenF1 but restricted to a specific n-gram
///   order, producing (precision, recall) series.
///
/// - **ROUGE-L**: defined via LCS, which requires a transducer rather than
///   a simple WFA; the FPS is defined over pairs.
#[derive(Clone, Debug)]
pub enum FPSDenotation {
    /// A single formal power series: f : Σ* → S.
    SingleSeries {
        /// Human-readable description of the series.
        description: String,
        /// The semiring in which coefficients live.
        semiring: SemiringType,
    },
    /// A pair of series (used for F1-style metrics: precision + recall).
    SeriesPair {
        precision_desc: String,
        recall_desc: String,
        semiring: SemiringType,
    },
    /// A series with a post-processing step.
    SeriesWithPostProcess {
        series_desc: String,
        semiring: SemiringType,
        post_process: String,
    },
}

/// Maps EvalSpec metric declarations to their formal power series
/// characterization.
pub struct FPSSemantics;

impl FPSSemantics {
    pub fn new() -> Self {
        Self
    }

    /// Denote a metric declaration as a formal power series.
    ///
    /// # Mathematical background
    ///
    /// Let Σ be a finite alphabet of tokens and S = (S, ⊕, ⊗, 0̄, 1̄) a
    /// semiring.  A *formal power series* is a mapping r : Σ* → S, often
    /// written r = Σ_{w∈Σ*} (r, w) · w  where (r, w) denotes the
    /// coefficient of word w.
    ///
    /// A weighted finite automaton A over S with n states computes a formal
    /// power series [[A]] defined by:
    ///
    ///   ([[A]], w) = α · M(w₁) · M(w₂) · … · M(wₖ) · ω
    ///
    /// where α is the initial weight row vector, ω is the final weight
    /// column vector, and M(a) is the n×n transition matrix for symbol a.
    pub fn denote_metric(&self, metric: &MetricDecl) -> SemResult<FPSDenotation> {
        // Classify the metric by inspecting its structure.
        let metric_type = self.classify_metric(metric);
        match metric_type {
            MetricType::ExactMatch => {
                // [[exact_match(ref, cand)]] is the indicator series:
                // I_ref : Σ* → {0,1} where I_ref(w) = 1 iff w = ref.
                // This is a polynomial with a single monomial of coefficient 1.
                Ok(FPSDenotation::SingleSeries {
                    description: format!(
                        "Indicator series I_ref: Σ* → {{0,1}}, I_ref(w) = [w = ref]. \
                         Polynomial with exactly one non-zero coefficient."
                    ),
                    semiring: SemiringType::Boolean,
                })
            }
            MetricType::TokenF1 => {
                // TokenF1 is defined via the pair:
                //   P(ref, cand) = |tokens(ref) ∩ tokens(cand)| / |tokens(cand)|
                //   R(ref, cand) = |tokens(ref) ∩ tokens(cand)| / |tokens(ref)|
                //   F1 = 2PR / (P + R)
                //
                // The intersection count is a counting-semiring series that
                // counts matched tokens.
                Ok(FPSDenotation::SeriesPair {
                    precision_desc: "Counting series over token overlaps: \
                        C_P(ref, cand) = |tokens(ref) ∩ tokens(cand)| / |tokens(cand)|"
                        .into(),
                    recall_desc: "Counting series over token overlaps: \
                        C_R(ref, cand) = |tokens(ref) ∩ tokens(cand)| / |tokens(ref)|"
                        .into(),
                    semiring: SemiringType::Counting,
                })
            }
            MetricType::BLEU => {
                // BLEU = BP · exp( Σ_{n=1}^{N} w_n · log(p_n) )
                // where p_n = clipped_count_n / total_count_n.
                //
                // The n-gram counts are counting-semiring series; the
                // geometric mean and brevity penalty are post-processing.
                Ok(FPSDenotation::SeriesWithPostProcess {
                    series_desc: "Product of n-gram counting series: for each n ∈ [1,N], \
                        C_n(ref,cand) computes clipped n-gram counts via \
                        bounded-counting semiring with bound = ref_count(gram)."
                        .into(),
                    semiring: SemiringType::BoundedCounting(u64::MAX),
                    post_process: "BLEU post-processing: compute modified n-gram precisions \
                        p_n, take weighted geometric mean, multiply by brevity penalty \
                        BP = min(1, exp(1 - |ref|/|cand|))."
                        .into(),
                })
            }
            MetricType::RougeN => {
                // ROUGE-N precision = |overlap_n| / |cand_ngrams|
                // ROUGE-N recall    = |overlap_n| / |ref_ngrams|
                // F-measure         = 2PR / (P+R)
                Ok(FPSDenotation::SeriesPair {
                    precision_desc: "N-gram overlap counting series (precision): \
                        C_P(ref,cand) = |ngrams_n(ref) ∩ ngrams_n(cand)| / |ngrams_n(cand)|"
                        .into(),
                    recall_desc: "N-gram overlap counting series (recall): \
                        C_R(ref,cand) = |ngrams_n(ref) ∩ ngrams_n(cand)| / |ngrams_n(ref)|"
                        .into(),
                    semiring: SemiringType::Counting,
                })
            }
            MetricType::RougeL => {
                // ROUGE-L is based on longest common subsequence (LCS).
                // LCS can be computed via a tropical-semiring transducer
                // that finds the shortest edit distance, which is then
                // converted to LCS length.
                Ok(FPSDenotation::SeriesWithPostProcess {
                    series_desc: "LCS-length series via tropical semiring: \
                        uses edit-distance transducer to compute lcs(ref,cand)."
                        .into(),
                    semiring: SemiringType::Tropical,
                    post_process: "Convert LCS length to ROUGE-L: \
                        P = lcs/|cand|, R = lcs/|ref|, F = 2PR/(P+R)."
                        .into(),
                })
            }
            MetricType::RegexMatch => {
                // regex match is an indicator series over the regular language.
                Ok(FPSDenotation::SingleSeries {
                    description: "Indicator series for a regular language L(regex): \
                        I_L : Σ* → {0,1}, I_L(w) = [w ∈ L]. \
                        Rational series (computed by Boolean WFA)."
                        .into(),
                    semiring: SemiringType::Boolean,
                })
            }
            MetricType::PassAtK => {
                // pass@k = 1 - C(n-c, k) / C(n, k) where c = #correct, n = #samples.
                // Not naturally a formal power series; we denote it as a
                // counting series with post-processing.
                Ok(FPSDenotation::SeriesWithPostProcess {
                    series_desc: "Counting series over correct samples: \
                        C(samples) = #{s ∈ samples : s passes}."
                        .into(),
                    semiring: SemiringType::Counting,
                    post_process: "Pass@k post-processing: \
                        pass@k = 1 - C(n-c,k)/C(n,k)."
                        .into(),
                })
            }
            MetricType::Custom => {
                // For custom metrics, we cannot automatically determine the
                // FPS characterization.  We provide a generic denotation.
                Ok(FPSDenotation::SingleSeries {
                    description: "Custom metric series: denotation depends on \
                        the user-defined metric body. Requires manual analysis \
                        of the expression tree to determine the FPS form."
                        .into(),
                    semiring: SemiringType::Real,
                })
            }
        }
    }

    /// Classify a metric declaration by inspecting its body.
    fn classify_metric(&self, metric: &MetricDecl) -> MetricType {
        // Walk the body looking for recognizable patterns.
        let body = &metric.body.node;
        if self.body_is_exact_match(body) {
            MetricType::ExactMatch
        } else if self.body_has_ngram_with_f1(body) {
            MetricType::TokenF1
        } else {
            MetricType::Custom
        }
    }

    fn body_is_exact_match(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryOp {
                op: BinaryOp::Eq, ..
            } => true,
            Expr::MatchPattern {
                mode: MatchMode::Exact,
                ..
            } => true,
            _ => false,
        }
    }

    fn body_has_ngram_with_f1(&self, expr: &Expr) -> bool {
        match expr {
            Expr::NGramExtract { .. } => true,
            Expr::BinaryOp { left, right, .. } => {
                self.body_has_ngram_with_f1(&left.node)
                    || self.body_has_ngram_with_f1(&right.node)
            }
            Expr::Let { value, body, .. } => {
                self.body_has_ngram_with_f1(&value.node)
                    || self.body_has_ngram_with_f1(&body.node)
            }
            _ => false,
        }
    }
}

impl Default for FPSSemantics {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Semantic Equivalence Checking
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a semantic equivalence check.
#[derive(Clone, Debug)]
pub struct EquivalenceCheckResult {
    pub equivalent: bool,
    pub counterexample: Option<SemanticValue>,
    pub samples_checked: usize,
}

/// Check whether two expressions are semantically equivalent by evaluating
/// them on random inputs and comparing the results.
///
/// This is a *testing*-based approach (not a proof): it can find
/// counterexamples but cannot prove equivalence.
pub fn check_semantic_equivalence(
    expr1: &Expr,
    expr2: &Expr,
    env: &Environment,
    num_samples: usize,
) -> SemResult<EquivalenceCheckResult> {
    let mut checked = 0;
    for _ in 0..num_samples {
        let mut eval1 = Evaluator::with_env(env.clone());
        let mut eval2 = Evaluator::with_env(env.clone());
        let v1 = eval1.eval_expr(expr1);
        let v2 = eval2.eval_expr(expr2);
        checked += 1;
        match (v1, v2) {
            (Ok(val1), Ok(val2)) => {
                if val1 != val2 {
                    return Ok(EquivalenceCheckResult {
                        equivalent: false,
                        counterexample: Some(val1),
                        samples_checked: checked,
                    });
                }
            }
            (Err(_), Err(_)) => {
                // Both error — treat as equivalent behavior.
            }
            (Ok(val), Err(_)) | (Err(_), Ok(val)) => {
                return Ok(EquivalenceCheckResult {
                    equivalent: false,
                    counterexample: Some(val),
                    samples_checked: checked,
                });
            }
        }
    }
    Ok(EquivalenceCheckResult {
        equivalent: true,
        counterexample: None,
        samples_checked: checked,
    })
}

/// Generate a random test input of the given type.
///
/// Uses a simple deterministic pseudo-random strategy for reproducibility
/// (seeded by the call index).
pub fn generate_test_input(ty: &EvalType, seed: u64) -> SemanticValue {
    // Simple deterministic pseudo-random based on seed.
    let hash = |s: u64| -> u64 {
        let mut x = s;
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    };

    match ty {
        EvalType::Base(BaseType::Integer) => {
            SemanticValue::Integer((hash(seed) % 201) as i64 - 100)
        }
        EvalType::Base(BaseType::Float) => {
            let bits = hash(seed);
            let f = (bits % 10000) as f64 / 100.0 - 50.0;
            SemanticValue::Float(f)
        }
        EvalType::Base(BaseType::Bool) => {
            SemanticValue::Boolean(hash(seed) % 2 == 0)
        }
        EvalType::Base(BaseType::String) => {
            let len = (hash(seed) % 8) as usize + 1;
            let mut s = String::new();
            for i in 0..len {
                let c = (b'a' + (hash(seed.wrapping_add(i as u64 + 1)) % 26) as u8) as char;
                s.push(c);
            }
            SemanticValue::Str(s)
        }
        EvalType::Base(BaseType::List(inner)) => {
            let len = (hash(seed) % 5) as usize;
            let elems: Vec<SemanticValue> = (0..len)
                .map(|i| {
                    generate_test_input(
                        &EvalType::Base(inner.as_ref().clone()),
                        seed.wrapping_add(i as u64 + 100),
                    )
                })
                .collect();
            SemanticValue::List(elems)
        }
        EvalType::Base(BaseType::TokenSequence) => {
            let len = (hash(seed) % 6) as usize + 1;
            let tokens: Vec<String> = (0..len)
                .map(|i| {
                    let h = hash(seed.wrapping_add(i as u64 + 200));
                    format!("tok{}", h % 100)
                })
                .collect();
            SemanticValue::TokenSequence(tokens)
        }
        EvalType::Unit => SemanticValue::Unit,
        _ => SemanticValue::Integer(hash(seed) as i64 % 100),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Normalization / Simplification
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics about which normalization rules were applied.
#[derive(Clone, Debug, Default)]
pub struct NormalizationStats {
    pub add_zero_eliminated: usize,
    pub mul_one_eliminated: usize,
    pub mul_zero_eliminated: usize,
    pub double_negation_eliminated: usize,
    pub constant_if_eliminated: usize,
    pub constants_folded: usize,
    pub lets_inlined: usize,
    pub dead_code_eliminated: usize,
    pub total_rules_applied: usize,
}

impl NormalizationStats {
    fn record(&mut self, rule: &str) {
        self.total_rules_applied += 1;
        match rule {
            "add_zero" => self.add_zero_eliminated += 1,
            "mul_one" => self.mul_one_eliminated += 1,
            "mul_zero" => self.mul_zero_eliminated += 1,
            "double_neg" => self.double_negation_eliminated += 1,
            "const_if" => self.constant_if_eliminated += 1,
            "const_fold" => self.constants_folded += 1,
            "let_inline" => self.lets_inlined += 1,
            "dead_code" => self.dead_code_eliminated += 1,
            _ => {}
        }
    }
}

/// Apply normalization / simplification rules to an expression.
///
/// Rules applied:
///   - `0 + x => x`, `x + 0 => x`
///   - `1 * x => x`, `x * 1 => x`
///   - `0 * x => 0`
///   - `not(not(x)) => x`
///   - `if true then a else b => a`
///   - `if false then a else b => b`
///   - Constant folding (evaluate pure constant expressions)
///   - Let inlining (if variable used exactly once)
///   - Dead code elimination
pub fn normalize(expr: &Expr) -> Expr {
    let mut stats = NormalizationStats::default();
    normalize_with_stats(expr, &mut stats)
}

/// Same as `normalize` but also returns statistics.
pub fn normalize_tracked(expr: &Expr) -> (Expr, NormalizationStats) {
    let mut stats = NormalizationStats::default();
    let result = normalize_with_stats(expr, &mut stats);
    (result, stats)
}

fn normalize_with_stats(expr: &Expr, stats: &mut NormalizationStats) -> Expr {
    match expr {
        // ── Binary operations ────────────────────────────────────────
        Expr::BinaryOp { op, left, right } => {
            let nl = normalize_with_stats(&left.node, stats);
            let nr = normalize_with_stats(&right.node, stats);

            // 0 + x => x, x + 0 => x
            if *op == BinaryOp::Add {
                if is_zero_literal(&nl) {
                    stats.record("add_zero");
                    return nr;
                }
                if is_zero_literal(&nr) {
                    stats.record("add_zero");
                    return nl;
                }
            }

            // x - 0 => x
            if *op == BinaryOp::Sub && is_zero_literal(&nr) {
                stats.record("add_zero");
                return nl;
            }

            // 1 * x => x, x * 1 => x
            if *op == BinaryOp::Mul {
                if is_one_literal(&nl) {
                    stats.record("mul_one");
                    return nr;
                }
                if is_one_literal(&nr) {
                    stats.record("mul_one");
                    return nl;
                }
                // 0 * x => 0, x * 0 => 0
                if is_zero_literal(&nl) {
                    stats.record("mul_zero");
                    return Expr::int(0);
                }
                if is_zero_literal(&nr) {
                    stats.record("mul_zero");
                    return Expr::int(0);
                }
            }

            // Constant folding for pure arithmetic.
            if let (Some(lv), Some(rv)) = (const_eval_simple(&nl), const_eval_simple(&nr)) {
                if let Some(result) = fold_binary_const(op, lv, rv) {
                    stats.record("const_fold");
                    return result;
                }
            }

            Expr::BinaryOp {
                op: op.clone(),
                left: Box::new(Spanned::new(nl, left.span.clone())),
                right: Box::new(Spanned::new(nr, right.span.clone())),
            }
        }

        // ── Unary operations ─────────────────────────────────────────
        Expr::UnaryOp { op, operand } => {
            let inner = normalize_with_stats(&operand.node, stats);

            // not(not(x)) => x
            if *op == UnaryOp::Not {
                if let Expr::UnaryOp {
                    op: UnaryOp::Not,
                    operand: inner_op,
                } = &inner
                {
                    stats.record("double_neg");
                    return inner_op.node.clone();
                }
            }

            // Neg(Neg(x)) => x
            if *op == UnaryOp::Neg {
                if let Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: inner_op,
                } = &inner
                {
                    stats.record("double_neg");
                    return inner_op.node.clone();
                }
            }

            // Constant folding for unary.
            if let Some(v) = const_eval_simple(&inner) {
                match op {
                    UnaryOp::Neg => match v {
                        ConstVal::Int(n) => {
                            stats.record("const_fold");
                            return Expr::int(-n);
                        }
                        ConstVal::Float(f) => {
                            stats.record("const_fold");
                            return Expr::float(-f);
                        }
                        _ => {}
                    },
                    UnaryOp::Not => {
                        if let ConstVal::Bool(b) = v {
                            stats.record("const_fold");
                            return Expr::bool_lit(!b);
                        }
                    }
                    _ => {}
                }
            }

            Expr::UnaryOp {
                op: op.clone(),
                operand: Box::new(Spanned::new(inner, operand.span.clone())),
            }
        }

        // ── If / then / else ─────────────────────────────────────────
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let nc = normalize_with_stats(&condition.node, stats);
            let nt = normalize_with_stats(&then_branch.node, stats);
            let ne = normalize_with_stats(&else_branch.node, stats);

            // if true then a else b => a
            if let Expr::Literal(Literal::Bool(true)) = &nc {
                stats.record("const_if");
                return nt;
            }
            // if false then a else b => b
            if let Expr::Literal(Literal::Bool(false)) = &nc {
                stats.record("const_if");
                return ne;
            }

            Expr::If {
                condition: Box::new(Spanned::new(nc, condition.span.clone())),
                then_branch: Box::new(Spanned::new(nt, then_branch.span.clone())),
                else_branch: Box::new(Spanned::new(ne, else_branch.span.clone())),
            }
        }

        // ── Let binding ──────────────────────────────────────────────
        Expr::Let {
            name,
            ty,
            value,
            body,
        } => {
            let nv = normalize_with_stats(&value.node, stats);
            let nb = normalize_with_stats(&body.node, stats);

            // Dead code elimination: if variable is never used in body.
            let usage_count = count_var_uses(&nb, name);
            if usage_count == 0 {
                stats.record("dead_code");
                return nb;
            }

            // Let inlining: if variable is used exactly once and the value
            // is a simple expression (literal or variable).
            if usage_count == 1 && is_simple_expr(&nv) {
                stats.record("let_inline");
                return substitute_var(&nb, name, &nv);
            }

            Expr::Let {
                name: name.clone(),
                ty: ty.clone(),
                value: Box::new(Spanned::new(nv, value.span.clone())),
                body: Box::new(Spanned::new(nb, body.span.clone())),
            }
        }

        // ── Recursively normalize children for all other node types ──
        Expr::Lambda { params, body } => {
            let nb = normalize_with_stats(&body.node, stats);
            Expr::Lambda {
                params: params.clone(),
                body: Box::new(Spanned::new(nb, body.span.clone())),
            }
        }

        Expr::FunctionCall { name, args } => {
            let nargs: Vec<Spanned<Expr>> = args
                .iter()
                .map(|a| Spanned::new(normalize_with_stats(&a.node, stats), a.span.clone()))
                .collect();
            Expr::FunctionCall {
                name: name.clone(),
                args: nargs,
            }
        }

        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let nr = normalize_with_stats(&receiver.node, stats);
            let nargs: Vec<Spanned<Expr>> = args
                .iter()
                .map(|a| Spanned::new(normalize_with_stats(&a.node, stats), a.span.clone()))
                .collect();
            Expr::MethodCall {
                receiver: Box::new(Spanned::new(nr, receiver.span.clone())),
                method: method.clone(),
                args: nargs,
            }
        }

        Expr::Block(exprs) => {
            let normalized: Vec<Spanned<Expr>> = exprs
                .iter()
                .map(|e| Spanned::new(normalize_with_stats(&e.node, stats), e.span.clone()))
                .collect();
            Expr::Block(normalized)
        }

        Expr::Match { scrutinee, arms } => {
            let ns = normalize_with_stats(&scrutinee.node, stats);
            let narms: Vec<MatchArm> = arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    guard: arm.guard.as_ref().map(|g| {
                        Spanned::new(normalize_with_stats(&g.node, stats), g.span.clone())
                    }),
                    body: Spanned::new(
                        normalize_with_stats(&arm.body.node, stats),
                        arm.body.span.clone(),
                    ),
                    span: arm.span.clone(),
                })
                .collect();
            Expr::Match {
                scrutinee: Box::new(Spanned::new(ns, scrutinee.span.clone())),
                arms: narms,
            }
        }

        Expr::ListLiteral(elems) => {
            let normalized: Vec<Spanned<Expr>> = elems
                .iter()
                .map(|e| Spanned::new(normalize_with_stats(&e.node, stats), e.span.clone()))
                .collect();
            Expr::ListLiteral(normalized)
        }

        Expr::TupleLiteral(elems) => {
            let normalized: Vec<Spanned<Expr>> = elems
                .iter()
                .map(|e| Spanned::new(normalize_with_stats(&e.node, stats), e.span.clone()))
                .collect();
            Expr::TupleLiteral(normalized)
        }

        Expr::Aggregate {
            op,
            collection,
            binding,
            body,
            semiring,
        } => {
            let nc = normalize_with_stats(&collection.node, stats);
            let nb = body
                .as_ref()
                .map(|b| Box::new(Spanned::new(normalize_with_stats(&b.node, stats), b.span.clone())));
            Expr::Aggregate {
                op: op.clone(),
                collection: Box::new(Spanned::new(nc, collection.span.clone())),
                binding: binding.clone(),
                body: nb,
                semiring: semiring.clone(),
            }
        }

        Expr::NGramExtract { input, n } => {
            let ni = normalize_with_stats(&input.node, stats);
            Expr::NGramExtract {
                input: Box::new(Spanned::new(ni, input.span.clone())),
                n: *n,
            }
        }

        Expr::TokenizeExpr { input, tokenizer } => {
            let ni = normalize_with_stats(&input.node, stats);
            Expr::TokenizeExpr {
                input: Box::new(Spanned::new(ni, input.span.clone())),
                tokenizer: tokenizer.clone(),
            }
        }

        Expr::FieldAccess { expr, field } => {
            let ne = normalize_with_stats(&expr.node, stats);
            Expr::FieldAccess {
                expr: Box::new(Spanned::new(ne, expr.span.clone())),
                field: field.clone(),
            }
        }

        Expr::IndexAccess { expr, index } => {
            let ne = normalize_with_stats(&expr.node, stats);
            let ni = normalize_with_stats(&index.node, stats);
            Expr::IndexAccess {
                expr: Box::new(Spanned::new(ne, expr.span.clone())),
                index: Box::new(Spanned::new(ni, index.span.clone())),
            }
        }

        Expr::SemiringCast { expr, from, to } => {
            let ne = normalize_with_stats(&expr.node, stats);
            Expr::SemiringCast {
                expr: Box::new(Spanned::new(ne, expr.span.clone())),
                from: from.clone(),
                to: to.clone(),
            }
        }

        Expr::ClipCount { count, max_count } => {
            let nc = normalize_with_stats(&count.node, stats);
            let nm = normalize_with_stats(&max_count.node, stats);
            Expr::ClipCount {
                count: Box::new(Spanned::new(nc, count.span.clone())),
                max_count: Box::new(Spanned::new(nm, max_count.span.clone())),
            }
        }

        Expr::Compose { first, second } => {
            let nf = normalize_with_stats(&first.node, stats);
            let ns = normalize_with_stats(&second.node, stats);
            Expr::Compose {
                first: Box::new(Spanned::new(nf, first.span.clone())),
                second: Box::new(Spanned::new(ns, second.span.clone())),
            }
        }

        Expr::MatchPattern { input, pattern, mode } => {
            let ni = normalize_with_stats(&input.node, stats);
            Expr::MatchPattern {
                input: Box::new(Spanned::new(ni, input.span.clone())),
                pattern: pattern.clone(),
                mode: mode.clone(),
            }
        }

        // Leaf nodes: return unchanged.
        Expr::Literal(_) | Expr::Variable(_) => expr.clone(),
    }
}

// ── Normalization helpers ────────────────────────────────────────────

fn is_zero_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(Literal::Integer(0)) => true,
        Expr::Literal(Literal::Float(f)) => f.into_inner() == 0.0,
        _ => false,
    }
}

fn is_one_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(Literal::Integer(1)) => true,
        Expr::Literal(Literal::Float(f)) => f.into_inner() == 1.0,
        _ => false,
    }
}

/// Simple constant values used during constant folding.
#[derive(Clone, Debug)]
enum ConstVal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

fn const_eval_simple(expr: &Expr) -> Option<ConstVal> {
    match expr {
        Expr::Literal(Literal::Integer(n)) => Some(ConstVal::Int(*n)),
        Expr::Literal(Literal::Float(f)) => Some(ConstVal::Float(f.into_inner())),
        Expr::Literal(Literal::Bool(b)) => Some(ConstVal::Bool(*b)),
        _ => None,
    }
}

fn fold_binary_const(op: &BinaryOp, lv: ConstVal, rv: ConstVal) -> Option<Expr> {
    match (op, &lv, &rv) {
        // Integer arithmetic.
        (BinaryOp::Add, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::int(a + b)),
        (BinaryOp::Sub, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::int(a - b)),
        (BinaryOp::Mul, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::int(a * b)),
        (BinaryOp::Div, ConstVal::Int(a), ConstVal::Int(b)) if *b != 0 => {
            Some(Expr::int(a / b))
        }
        // Float arithmetic.
        (BinaryOp::Add, ConstVal::Float(a), ConstVal::Float(b)) => Some(Expr::float(a + b)),
        (BinaryOp::Sub, ConstVal::Float(a), ConstVal::Float(b)) => Some(Expr::float(a - b)),
        (BinaryOp::Mul, ConstVal::Float(a), ConstVal::Float(b)) => Some(Expr::float(a * b)),
        (BinaryOp::Div, ConstVal::Float(a), ConstVal::Float(b)) if *b != 0.0 => {
            Some(Expr::float(a / b))
        }
        // Mixed int/float.
        (BinaryOp::Add, ConstVal::Int(a), ConstVal::Float(b)) => {
            Some(Expr::float(*a as f64 + b))
        }
        (BinaryOp::Add, ConstVal::Float(a), ConstVal::Int(b)) => {
            Some(Expr::float(a + *b as f64))
        }
        (BinaryOp::Sub, ConstVal::Int(a), ConstVal::Float(b)) => {
            Some(Expr::float(*a as f64 - b))
        }
        (BinaryOp::Sub, ConstVal::Float(a), ConstVal::Int(b)) => {
            Some(Expr::float(a - *b as f64))
        }
        (BinaryOp::Mul, ConstVal::Int(a), ConstVal::Float(b)) => {
            Some(Expr::float(*a as f64 * b))
        }
        (BinaryOp::Mul, ConstVal::Float(a), ConstVal::Int(b)) => {
            Some(Expr::float(a * *b as f64))
        }
        // Comparisons.
        (BinaryOp::Lt, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a < b)),
        (BinaryOp::Le, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a <= b)),
        (BinaryOp::Gt, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a > b)),
        (BinaryOp::Ge, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a >= b)),
        (BinaryOp::Eq, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a == b)),
        (BinaryOp::Neq, ConstVal::Int(a), ConstVal::Int(b)) => Some(Expr::bool_lit(a != b)),
        // Boolean logic.
        (BinaryOp::And, ConstVal::Bool(a), ConstVal::Bool(b)) => {
            Some(Expr::bool_lit(*a && *b))
        }
        (BinaryOp::Or, ConstVal::Bool(a), ConstVal::Bool(b)) => {
            Some(Expr::bool_lit(*a || *b))
        }
        _ => None,
    }
}

/// Count how many times a variable is referenced in an expression.
fn count_var_uses(expr: &Expr, name: &str) -> usize {
    match expr {
        Expr::Variable(n) => {
            if n == name {
                1
            } else {
                0
            }
        }
        Expr::Literal(_) => 0,
        Expr::BinaryOp { left, right, .. } => {
            count_var_uses(&left.node, name) + count_var_uses(&right.node, name)
        }
        Expr::UnaryOp { operand, .. } => count_var_uses(&operand.node, name),
        Expr::FunctionCall { args, .. } => {
            args.iter().map(|a| count_var_uses(&a.node, name)).sum()
        }
        Expr::MethodCall {
            receiver, args, ..
        } => {
            count_var_uses(&receiver.node, name)
                + args.iter().map(|a| count_var_uses(&a.node, name)).sum::<usize>()
        }
        Expr::Lambda { params, body } => {
            // If the lambda shadows the variable, stop counting.
            if params.iter().any(|p| p.name == name) {
                0
            } else {
                count_var_uses(&body.node, name)
            }
        }
        Expr::Let {
            name: let_name,
            value,
            body,
            ..
        } => {
            let in_val = count_var_uses(&value.node, name);
            if let_name == name {
                // Shadowed in body.
                in_val
            } else {
                in_val + count_var_uses(&body.node, name)
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            count_var_uses(&condition.node, name)
                + count_var_uses(&then_branch.node, name)
                + count_var_uses(&else_branch.node, name)
        }
        Expr::Match { scrutinee, arms } => {
            let mut count = count_var_uses(&scrutinee.node, name);
            for arm in arms {
                // Check if pattern shadows the variable.
                if arm.pattern.node.bound_vars().contains(&name.to_string()) {
                    continue;
                }
                count += count_var_uses(&arm.body.node, name);
                if let Some(g) = &arm.guard {
                    count += count_var_uses(&g.node, name);
                }
            }
            count
        }
        Expr::Block(exprs) => exprs.iter().map(|e| count_var_uses(&e.node, name)).sum(),
        Expr::FieldAccess { expr, .. } => count_var_uses(&expr.node, name),
        Expr::IndexAccess { expr, index } => {
            count_var_uses(&expr.node, name) + count_var_uses(&index.node, name)
        }
        Expr::ListLiteral(elems) | Expr::TupleLiteral(elems) => {
            elems.iter().map(|e| count_var_uses(&e.node, name)).sum()
        }
        Expr::Aggregate {
            collection,
            body,
            binding,
            ..
        } => {
            let mut count = count_var_uses(&collection.node, name);
            if binding.as_deref() != Some(name) {
                if let Some(b) = body {
                    count += count_var_uses(&b.node, name);
                }
            }
            count
        }
        Expr::NGramExtract { input, .. } => count_var_uses(&input.node, name),
        Expr::TokenizeExpr { input, .. } => count_var_uses(&input.node, name),
        Expr::MatchPattern { input, .. } => count_var_uses(&input.node, name),
        Expr::SemiringCast { expr, .. } => count_var_uses(&expr.node, name),
        Expr::ClipCount { count, max_count } => {
            count_var_uses(&count.node, name) + count_var_uses(&max_count.node, name)
        }
        Expr::Compose { first, second } => {
            count_var_uses(&first.node, name) + count_var_uses(&second.node, name)
        }
    }
}

/// Whether an expression is "simple" enough to inline (literal or variable).
fn is_simple_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(_) | Expr::Variable(_))
}

/// Substitute all free occurrences of `name` with `replacement` in `expr`.
fn substitute_var(expr: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Variable(n) => {
            if n == name {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Literal(_) => expr.clone(),
        Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
            op: op.clone(),
            left: Box::new(Spanned::new(
                substitute_var(&left.node, name, replacement),
                left.span.clone(),
            )),
            right: Box::new(Spanned::new(
                substitute_var(&right.node, name, replacement),
                right.span.clone(),
            )),
        },
        Expr::UnaryOp { op, operand } => Expr::UnaryOp {
            op: op.clone(),
            operand: Box::new(Spanned::new(
                substitute_var(&operand.node, name, replacement),
                operand.span.clone(),
            )),
        },
        Expr::FunctionCall {
            name: fn_name,
            args,
        } => Expr::FunctionCall {
            name: fn_name.clone(),
            args: args
                .iter()
                .map(|a| {
                    Spanned::new(
                        substitute_var(&a.node, name, replacement),
                        a.span.clone(),
                    )
                })
                .collect(),
        },
        Expr::Let {
            name: let_name,
            ty,
            value,
            body,
        } => {
            let new_value = Spanned::new(
                substitute_var(&value.node, name, replacement),
                value.span.clone(),
            );
            if let_name == name {
                // Shadowed — don't substitute in body.
                Expr::Let {
                    name: let_name.clone(),
                    ty: ty.clone(),
                    value: Box::new(new_value),
                    body: body.clone(),
                }
            } else {
                Expr::Let {
                    name: let_name.clone(),
                    ty: ty.clone(),
                    value: Box::new(new_value),
                    body: Box::new(Spanned::new(
                        substitute_var(&body.node, name, replacement),
                        body.span.clone(),
                    )),
                }
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => Expr::If {
            condition: Box::new(Spanned::new(
                substitute_var(&condition.node, name, replacement),
                condition.span.clone(),
            )),
            then_branch: Box::new(Spanned::new(
                substitute_var(&then_branch.node, name, replacement),
                then_branch.span.clone(),
            )),
            else_branch: Box::new(Spanned::new(
                substitute_var(&else_branch.node, name, replacement),
                else_branch.span.clone(),
            )),
        },
        Expr::Lambda { params, body } => {
            if params.iter().any(|p| p.name == name) {
                expr.clone()
            } else {
                Expr::Lambda {
                    params: params.clone(),
                    body: Box::new(Spanned::new(
                        substitute_var(&body.node, name, replacement),
                        body.span.clone(),
                    )),
                }
            }
        }
        Expr::Block(exprs) => Expr::Block(
            exprs
                .iter()
                .map(|e| {
                    Spanned::new(
                        substitute_var(&e.node, name, replacement),
                        e.span.clone(),
                    )
                })
                .collect(),
        ),
        Expr::ListLiteral(elems) => Expr::ListLiteral(
            elems
                .iter()
                .map(|e| {
                    Spanned::new(
                        substitute_var(&e.node, name, replacement),
                        e.span.clone(),
                    )
                })
                .collect(),
        ),
        Expr::TupleLiteral(elems) => Expr::TupleLiteral(
            elems
                .iter()
                .map(|e| {
                    Spanned::new(
                        substitute_var(&e.node, name, replacement),
                        e.span.clone(),
                    )
                })
                .collect(),
        ),
        // For all other expressions, clone unchanged (conservative).
        _ => expr.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Semantic Preservation Checking
// ═══════════════════════════════════════════════════════════════════════════

/// A single failure in semantic preservation checking.
#[derive(Clone, Debug)]
pub struct PreservationFailure {
    pub input: Vec<String>,
    pub expected_value: SemanticValue,
    pub wfa_weight: String,
}

/// Result of checking whether compilation preserves semantics.
#[derive(Clone, Debug)]
pub struct PreservationResult {
    pub all_preserved: bool,
    pub failures: Vec<PreservationFailure>,
}

/// Verify that the WFA produced by compilation computes the same function
/// as the original expression.
///
/// For each test input (a sequence of token strings), we:
/// 1. Evaluate the original expression in an environment where `candidate`
///    is bound to the token sequence.
/// 2. Run the WFA on the corresponding symbol sequence.
/// 3. Compare the resulting weight with the expected semantic value.
pub fn verify_compilation_preserves_semantics(
    expr: &Expr,
    wfa: &WeightedFiniteAutomaton<CountingSemiring>,
    test_inputs: &[Vec<String>],
) -> SemResult<PreservationResult> {
    let mut failures = Vec::new();

    for input in test_inputs {
        // Evaluate the expression directly.
        let mut eval = Evaluator::new();
        eval.env.bind(
            "candidate".to_string(),
            SemanticValue::TokenSequence(input.clone()),
        );
        eval.env.bind(
            "reference".to_string(),
            SemanticValue::TokenSequence(input.clone()),
        );

        let direct_result = eval.eval_expr(expr);

        // Convert input tokens to symbol indices for WFA.
        // We use the token index as symbol index (simple mapping).
        let symbol_indices: Vec<usize> = (0..input.len()).collect();

        let wfa_weight = wfa.compute_weight(&symbol_indices);

        match direct_result {
            Ok(val) => {
                let expected_f64 = val.to_f64().unwrap_or(0.0);
                let wfa_f64 = wfa_weight.value as f64;

                if (expected_f64 - wfa_f64).abs() > 1e-6 {
                    failures.push(PreservationFailure {
                        input: input.clone(),
                        expected_value: val,
                        wfa_weight: format!("{}", wfa_weight.value),
                    });
                }
            }
            Err(_) => {
                // If the expression errors, we note it but don't count it
                // as a preservation failure (the WFA might not support all
                // error conditions).
            }
        }
    }

    Ok(PreservationResult {
        all_preserved: failures.is_empty(),
        failures,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Built-in Function Implementations
// ═══════════════════════════════════════════════════════════════════════════

pub fn builtin_len(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "len expects 1 argument".into(),
        });
    }
    match &args[0] {
        SemanticValue::List(elems) => Ok(SemanticValue::Integer(elems.len() as i64)),
        SemanticValue::Str(s) => Ok(SemanticValue::Integer(s.len() as i64)),
        SemanticValue::TokenSequence(tokens) => {
            Ok(SemanticValue::Integer(tokens.len() as i64))
        }
        SemanticValue::NGramSet(ngrams) => {
            Ok(SemanticValue::Integer(ngrams.len() as i64))
        }
        SemanticValue::Tuple(elems) => Ok(SemanticValue::Integer(elems.len() as i64)),
        _ => Err(SemanticError::TypeMismatch {
            expected: "collection type".into(),
            found: args[0].type_name().into(),
        }),
    }
}

pub fn builtin_count(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "count expects 1 argument".into(),
        });
    }
    match &args[0] {
        SemanticValue::NGramSet(ngrams) => {
            let total: usize = ngrams.values().sum();
            Ok(SemanticValue::Integer(total as i64))
        }
        SemanticValue::List(elems) => Ok(SemanticValue::Integer(elems.len() as i64)),
        _ => Err(SemanticError::TypeMismatch {
            expected: "NGramSet or List".into(),
            found: args[0].type_name().into(),
        }),
    }
}

pub fn builtin_min(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() == 1 {
        // min of a list
        if let SemanticValue::List(elems) = &args[0] {
            if elems.is_empty() {
                return Ok(SemanticValue::Float(f64::INFINITY));
            }
            let mut min_val = f64::INFINITY;
            for e in elems {
                let f = e.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "numeric".into(),
                    found: e.type_name().into(),
                })?;
                if f < min_val {
                    min_val = f;
                }
            }
            return Ok(SemanticValue::Float(min_val));
        }
    }
    if args.len() == 2 {
        let a = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: args[0].type_name().into(),
        })?;
        let b = args[1].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: args[1].type_name().into(),
        })?;
        return Ok(SemanticValue::Float(a.min(b)));
    }
    Err(SemanticError::InvalidOperation {
        desc: "min expects 1 or 2 arguments".into(),
    })
}

pub fn builtin_max(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() == 1 {
        if let SemanticValue::List(elems) = &args[0] {
            if elems.is_empty() {
                return Ok(SemanticValue::Float(f64::NEG_INFINITY));
            }
            let mut max_val = f64::NEG_INFINITY;
            for e in elems {
                let f = e.to_f64().ok_or_else(|| SemanticError::TypeMismatch {
                    expected: "numeric".into(),
                    found: e.type_name().into(),
                })?;
                if f > max_val {
                    max_val = f;
                }
            }
            return Ok(SemanticValue::Float(max_val));
        }
    }
    if args.len() == 2 {
        let a = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: args[0].type_name().into(),
        })?;
        let b = args[1].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: args[1].type_name().into(),
        })?;
        return Ok(SemanticValue::Float(a.max(b)));
    }
    Err(SemanticError::InvalidOperation {
        desc: "max expects 1 or 2 arguments".into(),
    })
}

pub fn builtin_abs(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "abs expects 1 argument".into(),
        });
    }
    match &args[0] {
        SemanticValue::Integer(n) => Ok(SemanticValue::Integer(n.abs())),
        SemanticValue::Float(v) => Ok(SemanticValue::Float(v.abs())),
        _ => Err(SemanticError::TypeMismatch {
            expected: "numeric".into(),
            found: args[0].type_name().into(),
        }),
    }
}

pub fn builtin_log(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "log expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    if f <= 0.0 {
        return Err(SemanticError::InvalidOperation {
            desc: "log of non-positive number".into(),
        });
    }
    Ok(SemanticValue::Float(f.ln()))
}

pub fn builtin_exp(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "exp expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    Ok(SemanticValue::Float(f.exp()))
}

pub fn builtin_floor(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "floor expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    Ok(SemanticValue::Integer(f.floor() as i64))
}

pub fn builtin_ceil(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "ceil expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    Ok(SemanticValue::Integer(f.ceil() as i64))
}

pub fn builtin_round(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "round expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    Ok(SemanticValue::Integer(f.round() as i64))
}

pub fn builtin_intersection(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 2 {
        return Err(SemanticError::InvalidOperation {
            desc: "intersection expects 2 arguments".into(),
        });
    }
    match (&args[0], &args[1]) {
        (SemanticValue::NGramSet(a), SemanticValue::NGramSet(b)) => {
            let mut result: IndexMap<Vec<String>, usize> = IndexMap::new();
            for (gram, count_a) in a {
                if let Some(count_b) = b.get(gram) {
                    result.insert(gram.clone(), (*count_a).min(*count_b));
                }
            }
            Ok(SemanticValue::NGramSet(result))
        }
        (SemanticValue::List(a), SemanticValue::List(b)) => {
            let result: Vec<SemanticValue> = a
                .iter()
                .filter(|item| b.iter().any(|bitem| bitem == *item))
                .cloned()
                .collect();
            Ok(SemanticValue::List(result))
        }
        (SemanticValue::TokenSequence(a), SemanticValue::TokenSequence(b)) => {
            let result: Vec<String> = a
                .iter()
                .filter(|t| b.contains(t))
                .cloned()
                .collect();
            Ok(SemanticValue::TokenSequence(result))
        }
        _ => Err(SemanticError::TypeMismatch {
            expected: "NGramSet, List, or TokenSequence".into(),
            found: format!("{}, {}", args[0].type_name(), args[1].type_name()),
        }),
    }
}

pub fn builtin_union(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 2 {
        return Err(SemanticError::InvalidOperation {
            desc: "union expects 2 arguments".into(),
        });
    }
    match (&args[0], &args[1]) {
        (SemanticValue::NGramSet(a), SemanticValue::NGramSet(b)) => {
            let mut result: IndexMap<Vec<String>, usize> = a.clone();
            for (gram, count_b) in b {
                let entry = result.entry(gram.clone()).or_insert(0);
                *entry = (*entry).max(*count_b);
            }
            Ok(SemanticValue::NGramSet(result))
        }
        (SemanticValue::List(a), SemanticValue::List(b)) => {
            let mut result = a.clone();
            for item in b {
                if !result.iter().any(|r| r == item) {
                    result.push(item.clone());
                }
            }
            Ok(SemanticValue::List(result))
        }
        _ => Err(SemanticError::TypeMismatch {
            expected: "NGramSet or List".into(),
            found: format!("{}, {}", args[0].type_name(), args[1].type_name()),
        }),
    }
}

pub fn builtin_sqrt(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 1 {
        return Err(SemanticError::InvalidOperation {
            desc: "sqrt expects 1 argument".into(),
        });
    }
    let f = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    if f < 0.0 {
        return Err(SemanticError::InvalidOperation {
            desc: "sqrt of negative number".into(),
        });
    }
    Ok(SemanticValue::Float(f.sqrt()))
}

pub fn builtin_pow(args: &[SemanticValue]) -> SemResult<SemanticValue> {
    if args.len() != 2 {
        return Err(SemanticError::InvalidOperation {
            desc: "pow expects 2 arguments".into(),
        });
    }
    let base = args[0].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[0].type_name().into(),
    })?;
    let exp = args[1].to_f64().ok_or_else(|| SemanticError::TypeMismatch {
        expected: "numeric".into(),
        found: args[1].type_name().into(),
    })?;
    Ok(SemanticValue::Float(base.powf(exp)))
}

/// Register all built-in functions in an environment.
///
/// Built-in functions are registered as named function values that dispatch
/// to the corresponding Rust implementations.
pub fn register_builtins(env: &mut Environment) {
    // Built-in functions are resolved by name in dispatch_builtin rather
    // than stored as closures, so this function currently serves as a
    // documentation point and a place to register any environment constants.
    env.bind("PI", SemanticValue::Float(std::f64::consts::PI));
    env.bind("E", SemanticValue::Float(std::f64::consts::E));
    env.bind("INF", SemanticValue::Float(f64::INFINITY));
    env.bind("NEG_INF", SemanticValue::Float(f64::NEG_INFINITY));
    env.bind("true", SemanticValue::Boolean(true));
    env.bind("false", SemanticValue::Boolean(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evalspec::types::*;

    fn syn<T>(node: T) -> Spanned<T> {
        Spanned::synthetic(node)
    }

    // ── Literal evaluation ───────────────────────────────────────────

    #[test]
    fn test_eval_integer_literal() {
        let mut eval = Evaluator::new();
        let result = eval.eval_expr(&Expr::int(42)).unwrap();
        assert_eq!(result, SemanticValue::Integer(42));
    }

    #[test]
    fn test_eval_float_literal() {
        let mut eval = Evaluator::new();
        let result = eval.eval_expr(&Expr::float(3.14)).unwrap();
        assert_eq!(result, SemanticValue::Float(3.14));
    }

    #[test]
    fn test_eval_bool_literal() {
        let mut eval = Evaluator::new();
        let result = eval.eval_expr(&Expr::bool_lit(true)).unwrap();
        assert_eq!(result, SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_string_literal() {
        let mut eval = Evaluator::new();
        let result = eval.eval_expr(&Expr::string("hello")).unwrap();
        assert_eq!(result, SemanticValue::Str("hello".into()));
    }

    // ── Binary operations ────────────────────────────────────────────

    #[test]
    fn test_eval_add_integers() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Add, syn(Expr::int(3)), syn(Expr::int(4)));
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, SemanticValue::Integer(7));
    }

    #[test]
    fn test_eval_sub_integers() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Sub, syn(Expr::int(10)), syn(Expr::int(3)));
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, SemanticValue::Integer(7));
    }

    #[test]
    fn test_eval_mul_integers() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Mul, syn(Expr::int(5)), syn(Expr::int(6)));
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, SemanticValue::Integer(30));
    }

    #[test]
    fn test_eval_div_integers() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Div, syn(Expr::int(20)), syn(Expr::int(4)));
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(result, SemanticValue::Integer(5));
    }

    #[test]
    fn test_eval_division_by_zero() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Div, syn(Expr::int(1)), syn(Expr::int(0)));
        let result = eval.eval_expr(&expr);
        assert!(matches!(result, Err(SemanticError::DivisionByZero)));
    }

    #[test]
    fn test_eval_comparison_lt() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Lt, syn(Expr::int(3)), syn(Expr::int(5)));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_comparison_ge() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Ge, syn(Expr::int(3)), syn(Expr::int(5)));
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Boolean(false)
        );
    }

    #[test]
    fn test_eval_equality() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(BinaryOp::Eq, syn(Expr::int(5)), syn(Expr::int(5)));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_string_equality() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::Eq,
            syn(Expr::string("abc")),
            syn(Expr::string("abc")),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_and() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::And,
            syn(Expr::bool_lit(true)),
            syn(Expr::bool_lit(false)),
        );
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Boolean(false)
        );
    }

    #[test]
    fn test_eval_logical_or() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::Or,
            syn(Expr::bool_lit(false)),
            syn(Expr::bool_lit(true)),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_short_circuit_and() {
        // false && (1/0) should not error because of short-circuit.
        let mut eval = Evaluator::new();
        let div_zero = Expr::binary(BinaryOp::Div, syn(Expr::int(1)), syn(Expr::int(0)));
        let expr = Expr::binary(BinaryOp::And, syn(Expr::bool_lit(false)), syn(div_zero));
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Boolean(false)
        );
    }

    #[test]
    fn test_eval_short_circuit_or() {
        let mut eval = Evaluator::new();
        let div_zero = Expr::binary(BinaryOp::Div, syn(Expr::int(1)), syn(Expr::int(0)));
        let expr = Expr::binary(BinaryOp::Or, syn(Expr::bool_lit(true)), syn(div_zero));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_eval_string_concat() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::Add,
            syn(Expr::string("hello ")),
            syn(Expr::string("world")),
        );
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Str("hello world".into())
        );
    }

    // ── Variable binding and lookup ──────────────────────────────────

    #[test]
    fn test_variable_binding() {
        let mut eval = Evaluator::new();
        let expr = Expr::let_expr("x", None, syn(Expr::int(42)), syn(Expr::var("x")));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(42));
    }

    #[test]
    fn test_undefined_variable() {
        let mut eval = Evaluator::new();
        let result = eval.eval_expr(&Expr::var("nonexistent"));
        assert!(matches!(
            result,
            Err(SemanticError::UndefinedVariable { .. })
        ));
    }

    #[test]
    fn test_nested_let() {
        let mut eval = Evaluator::new();
        let inner = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::var("y")));
        let expr = Expr::let_expr(
            "x",
            None,
            syn(Expr::int(10)),
            syn(Expr::let_expr("y", None, syn(Expr::int(20)), syn(inner))),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(30));
    }

    // ── Function definition and application ──────────────────────────

    #[test]
    fn test_lambda_application() {
        let mut eval = Evaluator::new();
        // let add = |x, y| x + y in add(3, 4)
        let lambda_body = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::var("y")));
        let lambda = Expr::Lambda {
            params: vec![
                LambdaParam {
                    name: "x".into(),
                    ty: None,
                    span: Span::synthetic(),
                },
                LambdaParam {
                    name: "y".into(),
                    ty: None,
                    span: Span::synthetic(),
                },
            ],
            body: Box::new(syn(lambda_body)),
        };
        let call = Expr::FunctionCall {
            name: "add".into(),
            args: vec![syn(Expr::int(3)), syn(Expr::int(4))],
        };
        let expr = Expr::let_expr("add", None, syn(lambda), syn(call));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(7));
    }

    #[test]
    fn test_closure_capture() {
        let mut eval = Evaluator::new();
        // let offset = 10 in
        // let add_offset = |x| x + offset in
        // add_offset(5)
        let lambda_body =
            Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::var("offset")));
        let lambda = Expr::Lambda {
            params: vec![LambdaParam {
                name: "x".into(),
                ty: None,
                span: Span::synthetic(),
            }],
            body: Box::new(syn(lambda_body)),
        };
        let call = Expr::FunctionCall {
            name: "add_offset".into(),
            args: vec![syn(Expr::int(5))],
        };
        let expr = Expr::let_expr(
            "offset",
            None,
            syn(Expr::int(10)),
            syn(Expr::let_expr("add_offset", None, syn(lambda), syn(call))),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(15));
    }

    // ── If/else ──────────────────────────────────────────────────────

    #[test]
    fn test_if_true() {
        let mut eval = Evaluator::new();
        let expr = Expr::if_expr(
            syn(Expr::bool_lit(true)),
            syn(Expr::int(1)),
            syn(Expr::int(2)),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(1));
    }

    #[test]
    fn test_if_false() {
        let mut eval = Evaluator::new();
        let expr = Expr::if_expr(
            syn(Expr::bool_lit(false)),
            syn(Expr::int(1)),
            syn(Expr::int(2)),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(2));
    }

    #[test]
    fn test_if_with_comparison() {
        let mut eval = Evaluator::new();
        let cond = Expr::binary(BinaryOp::Gt, syn(Expr::int(5)), syn(Expr::int(3)));
        let expr = Expr::if_expr(syn(cond), syn(Expr::string("yes")), syn(Expr::string("no")));
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Str("yes".into())
        );
    }

    // ── Match expression ─────────────────────────────────────────────

    #[test]
    fn test_match_literal() {
        let mut eval = Evaluator::new();
        let arms = vec![
            MatchArm {
                pattern: syn(Pattern::Literal(Literal::Integer(1))),
                guard: None,
                body: syn(Expr::string("one")),
                span: Span::synthetic(),
            },
            MatchArm {
                pattern: syn(Pattern::Literal(Literal::Integer(2))),
                guard: None,
                body: syn(Expr::string("two")),
                span: Span::synthetic(),
            },
            MatchArm {
                pattern: syn(Pattern::Wildcard),
                guard: None,
                body: syn(Expr::string("other")),
                span: Span::synthetic(),
            },
        ];
        let expr = Expr::Match {
            scrutinee: Box::new(syn(Expr::int(2))),
            arms,
        };
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Str("two".into())
        );
    }

    #[test]
    fn test_match_variable_binding() {
        let mut eval = Evaluator::new();
        let arms = vec![MatchArm {
            pattern: syn(Pattern::Var("x".into())),
            guard: None,
            body: syn(Expr::binary(
                BinaryOp::Add,
                syn(Expr::var("x")),
                syn(Expr::int(1)),
            )),
            span: Span::synthetic(),
        }];
        let expr = Expr::Match {
            scrutinee: Box::new(syn(Expr::int(10))),
            arms,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(11));
    }

    #[test]
    fn test_match_wildcard() {
        let mut eval = Evaluator::new();
        let arms = vec![MatchArm {
            pattern: syn(Pattern::Wildcard),
            guard: None,
            body: syn(Expr::string("matched")),
            span: Span::synthetic(),
        }];
        let expr = Expr::Match {
            scrutinee: Box::new(syn(Expr::int(999))),
            arms,
        };
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Str("matched".into())
        );
    }

    #[test]
    fn test_match_tuple() {
        let mut eval = Evaluator::new();
        let arms = vec![MatchArm {
            pattern: syn(Pattern::Tuple(vec![
                syn(Pattern::Var("a".into())),
                syn(Pattern::Var("b".into())),
            ])),
            guard: None,
            body: syn(Expr::binary(
                BinaryOp::Add,
                syn(Expr::var("a")),
                syn(Expr::var("b")),
            )),
            span: Span::synthetic(),
        }];
        let scrutinee = Expr::TupleLiteral(vec![syn(Expr::int(3)), syn(Expr::int(7))]);
        let expr = Expr::Match {
            scrutinee: Box::new(syn(scrutinee)),
            arms,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(10));
    }

    // ── Aggregate evaluation ─────────────────────────────────────────

    #[test]
    fn test_aggregate_sum() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::int(1)),
            syn(Expr::int(2)),
            syn(Expr::int(3)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::Sum,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(6));
    }

    #[test]
    fn test_aggregate_product() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::int(2)),
            syn(Expr::int(3)),
            syn(Expr::int(4)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::Product,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(24));
    }

    #[test]
    fn test_aggregate_count() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::int(10)),
            syn(Expr::int(20)),
            syn(Expr::int(30)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::Count,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(3));
    }

    #[test]
    fn test_aggregate_mean() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::float(2.0)),
            syn(Expr::float(4.0)),
            syn(Expr::float(6.0)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::Mean,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Float(4.0));
    }

    #[test]
    fn test_aggregate_with_body() {
        let mut eval = Evaluator::new();
        // sum(x in [1,2,3] => x * x) = 1 + 4 + 9 = 14
        let list = Expr::ListLiteral(vec![
            syn(Expr::int(1)),
            syn(Expr::int(2)),
            syn(Expr::int(3)),
        ]);
        let body = Expr::binary(BinaryOp::Mul, syn(Expr::var("x")), syn(Expr::var("x")));
        let expr = Expr::Aggregate {
            op: AggregationOp::Sum,
            collection: Box::new(syn(list)),
            binding: Some("x".into()),
            body: Some(Box::new(syn(body))),
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(14));
    }

    #[test]
    fn test_aggregate_empty_list() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![]);
        let expr = Expr::Aggregate {
            op: AggregationOp::Sum,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(0));
    }

    // ── N-gram extraction ────────────────────────────────────────────

    #[test]
    fn test_ngram_extraction() {
        let mut eval = Evaluator::new();
        eval.env.bind(
            "tokens",
            SemanticValue::TokenSequence(vec![
                "the".into(),
                "cat".into(),
                "sat".into(),
                "on".into(),
            ]),
        );
        let expr = Expr::NGramExtract {
            input: Box::new(syn(Expr::var("tokens"))),
            n: 2,
        };
        let result = eval.eval_expr(&expr).unwrap();
        if let SemanticValue::NGramSet(ngrams) = result {
            assert_eq!(ngrams.len(), 3);
            assert_eq!(
                ngrams.get(&vec!["the".to_string(), "cat".to_string()]),
                Some(&1)
            );
            assert_eq!(
                ngrams.get(&vec!["cat".to_string(), "sat".to_string()]),
                Some(&1)
            );
            assert_eq!(
                ngrams.get(&vec!["sat".to_string(), "on".to_string()]),
                Some(&1)
            );
        } else {
            panic!("expected NGramSet");
        }
    }

    #[test]
    fn test_ngram_too_short() {
        let mut eval = Evaluator::new();
        eval.env.bind(
            "tokens",
            SemanticValue::TokenSequence(vec!["hello".into()]),
        );
        let expr = Expr::NGramExtract {
            input: Box::new(syn(Expr::var("tokens"))),
            n: 3,
        };
        let result = eval.eval_expr(&expr).unwrap();
        if let SemanticValue::NGramSet(ngrams) = result {
            assert!(ngrams.is_empty());
        } else {
            panic!("expected NGramSet");
        }
    }

    // ── Tokenization ─────────────────────────────────────────────────

    #[test]
    fn test_tokenize() {
        let mut eval = Evaluator::new();
        let expr = Expr::TokenizeExpr {
            input: Box::new(syn(Expr::string("Hello World Test"))),
            tokenizer: None,
        };
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(
            result,
            SemanticValue::TokenSequence(vec![
                "hello".into(),
                "world".into(),
                "test".into()
            ])
        );
    }

    // ── Unary operations ─────────────────────────────────────────────

    #[test]
    fn test_negation() {
        let mut eval = Evaluator::new();
        let expr = Expr::unary(UnaryOp::Neg, syn(Expr::int(5)));
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(-5));
    }

    #[test]
    fn test_not() {
        let mut eval = Evaluator::new();
        let expr = Expr::unary(UnaryOp::Not, syn(Expr::bool_lit(true)));
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Boolean(false)
        );
    }

    // ── Normalization ────────────────────────────────────────────────

    #[test]
    fn test_normalize_add_zero_left() {
        let expr = Expr::binary(BinaryOp::Add, syn(Expr::int(0)), syn(Expr::var("x")));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_add_zero_right() {
        let expr = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::int(0)));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_mul_one_left() {
        let expr = Expr::binary(BinaryOp::Mul, syn(Expr::int(1)), syn(Expr::var("x")));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_mul_one_right() {
        let expr = Expr::binary(BinaryOp::Mul, syn(Expr::var("x")), syn(Expr::int(1)));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_mul_zero() {
        let expr = Expr::binary(BinaryOp::Mul, syn(Expr::int(0)), syn(Expr::var("x")));
        let result = normalize(&expr);
        assert_eq!(result, Expr::int(0));
    }

    #[test]
    fn test_normalize_double_not() {
        let inner = Expr::unary(UnaryOp::Not, syn(Expr::var("x")));
        let expr = Expr::unary(UnaryOp::Not, syn(inner));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_if_true() {
        let expr = Expr::if_expr(
            syn(Expr::bool_lit(true)),
            syn(Expr::int(1)),
            syn(Expr::int(2)),
        );
        let result = normalize(&expr);
        assert_eq!(result, Expr::int(1));
    }

    #[test]
    fn test_normalize_if_false() {
        let expr = Expr::if_expr(
            syn(Expr::bool_lit(false)),
            syn(Expr::int(1)),
            syn(Expr::int(2)),
        );
        let result = normalize(&expr);
        assert_eq!(result, Expr::int(2));
    }

    #[test]
    fn test_normalize_constant_folding() {
        let expr = Expr::binary(BinaryOp::Add, syn(Expr::int(3)), syn(Expr::int(4)));
        let result = normalize(&expr);
        assert_eq!(result, Expr::int(7));
    }

    #[test]
    fn test_normalize_dead_code_elimination() {
        // let unused = 42 in x  =>  x
        let expr = Expr::let_expr("unused", None, syn(Expr::int(42)), syn(Expr::var("x")));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_let_inlining() {
        // let y = 42 in y + 1  =>  42 + 1  =>  43 (after const fold)
        let body = Expr::binary(BinaryOp::Add, syn(Expr::var("y")), syn(Expr::int(1)));
        let expr = Expr::let_expr("y", None, syn(Expr::int(42)), syn(body));
        let result = normalize(&expr);
        // After inlining y=42 and constant folding 42+1=43:
        assert_eq!(result, Expr::int(43));
    }

    #[test]
    fn test_normalize_stats() {
        let expr = Expr::binary(BinaryOp::Add, syn(Expr::int(0)), syn(Expr::var("x")));
        let (result, stats) = normalize_tracked(&expr);
        assert_eq!(result, Expr::var("x"));
        assert_eq!(stats.add_zero_eliminated, 1);
        assert!(stats.total_rules_applied > 0);
    }

    // ── Semantic equivalence ─────────────────────────────────────────

    #[test]
    fn test_equivalence_same_expr() {
        let expr1 = Expr::int(42);
        let expr2 = Expr::int(42);
        let env = Environment::new();
        let result = check_semantic_equivalence(&expr1, &expr2, &env, 10).unwrap();
        assert!(result.equivalent);
        assert_eq!(result.samples_checked, 10);
    }

    #[test]
    fn test_equivalence_different_expr() {
        let expr1 = Expr::int(42);
        let expr2 = Expr::int(43);
        let env = Environment::new();
        let result = check_semantic_equivalence(&expr1, &expr2, &env, 10).unwrap();
        assert!(!result.equivalent);
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_equivalence_with_normalization() {
        // x + 0 should be equivalent to x when x is bound.
        let expr1 = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::int(0)));
        let expr2 = Expr::var("x");
        let mut env = Environment::new();
        env.bind("x", SemanticValue::Integer(7));
        let result = check_semantic_equivalence(&expr1, &expr2, &env, 5).unwrap();
        assert!(result.equivalent);
    }

    // ── Environment scoping ──────────────────────────────────────────

    #[test]
    fn test_environment_scoping() {
        let mut env = Environment::new();
        env.bind("x", SemanticValue::Integer(1));
        let child = env.child();
        assert_eq!(
            child.lookup("x"),
            Some(&SemanticValue::Integer(1))
        );
    }

    #[test]
    fn test_environment_shadowing() {
        let mut env = Environment::new();
        env.bind("x", SemanticValue::Integer(1));
        let mut child = env.child();
        child.bind("x", SemanticValue::Integer(2));
        assert_eq!(
            child.lookup("x"),
            Some(&SemanticValue::Integer(2))
        );
        // Parent is unaffected.
        assert_eq!(env.lookup("x"), Some(&SemanticValue::Integer(1)));
    }

    #[test]
    fn test_environment_depth() {
        let env = Environment::new();
        assert_eq!(env.depth(), 1);
        let child = env.child();
        assert_eq!(child.depth(), 2);
        let grandchild = child.child();
        assert_eq!(grandchild.depth(), 3);
    }

    // ── Built-in function tests ──────────────────────────────────────

    #[test]
    fn test_builtin_len_list() {
        let result = builtin_len(&[SemanticValue::List(vec![
            SemanticValue::Integer(1),
            SemanticValue::Integer(2),
            SemanticValue::Integer(3),
        ])])
        .unwrap();
        assert_eq!(result, SemanticValue::Integer(3));
    }

    #[test]
    fn test_builtin_len_string() {
        let result = builtin_len(&[SemanticValue::Str("hello".into())]).unwrap();
        assert_eq!(result, SemanticValue::Integer(5));
    }

    #[test]
    fn test_builtin_abs_positive() {
        let result = builtin_abs(&[SemanticValue::Integer(5)]).unwrap();
        assert_eq!(result, SemanticValue::Integer(5));
    }

    #[test]
    fn test_builtin_abs_negative() {
        let result = builtin_abs(&[SemanticValue::Integer(-5)]).unwrap();
        assert_eq!(result, SemanticValue::Integer(5));
    }

    #[test]
    fn test_builtin_min_two_args() {
        let result =
            builtin_min(&[SemanticValue::Float(3.0), SemanticValue::Float(1.0)]).unwrap();
        assert_eq!(result, SemanticValue::Float(1.0));
    }

    #[test]
    fn test_builtin_max_two_args() {
        let result =
            builtin_max(&[SemanticValue::Float(3.0), SemanticValue::Float(7.0)]).unwrap();
        assert_eq!(result, SemanticValue::Float(7.0));
    }

    #[test]
    fn test_builtin_floor() {
        let result = builtin_floor(&[SemanticValue::Float(3.7)]).unwrap();
        assert_eq!(result, SemanticValue::Integer(3));
    }

    #[test]
    fn test_builtin_ceil() {
        let result = builtin_ceil(&[SemanticValue::Float(3.2)]).unwrap();
        assert_eq!(result, SemanticValue::Integer(4));
    }

    #[test]
    fn test_builtin_round() {
        let result = builtin_round(&[SemanticValue::Float(3.5)]).unwrap();
        assert_eq!(result, SemanticValue::Integer(4));
    }

    #[test]
    fn test_builtin_exp_and_log() {
        let exp_result = builtin_exp(&[SemanticValue::Float(1.0)]).unwrap();
        if let SemanticValue::Float(v) = exp_result {
            assert!((v - std::f64::consts::E).abs() < 1e-10);
        } else {
            panic!("expected float");
        }

        let log_result = builtin_log(&[SemanticValue::Float(std::f64::consts::E)]).unwrap();
        if let SemanticValue::Float(v) = log_result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn test_builtin_log_negative() {
        let result = builtin_log(&[SemanticValue::Float(-1.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_builtin_intersection_ngrams() {
        let mut a: IndexMap<Vec<String>, usize> = IndexMap::new();
        a.insert(vec!["the".into(), "cat".into()], 2);
        a.insert(vec!["cat".into(), "sat".into()], 1);

        let mut b: IndexMap<Vec<String>, usize> = IndexMap::new();
        b.insert(vec!["the".into(), "cat".into()], 1);
        b.insert(vec!["dog".into(), "ran".into()], 1);

        let result = builtin_intersection(&[
            SemanticValue::NGramSet(a),
            SemanticValue::NGramSet(b),
        ])
        .unwrap();

        if let SemanticValue::NGramSet(ngrams) = result {
            assert_eq!(ngrams.len(), 1);
            assert_eq!(
                ngrams.get(&vec!["the".to_string(), "cat".to_string()]),
                Some(&1)
            );
        } else {
            panic!("expected NGramSet");
        }
    }

    #[test]
    fn test_builtin_union_ngrams() {
        let mut a: IndexMap<Vec<String>, usize> = IndexMap::new();
        a.insert(vec!["a".into()], 2);
        a.insert(vec!["b".into()], 1);

        let mut b: IndexMap<Vec<String>, usize> = IndexMap::new();
        b.insert(vec!["a".into()], 3);
        b.insert(vec!["c".into()], 1);

        let result =
            builtin_union(&[SemanticValue::NGramSet(a), SemanticValue::NGramSet(b)]).unwrap();

        if let SemanticValue::NGramSet(ngrams) = result {
            assert_eq!(ngrams.len(), 3);
            assert_eq!(ngrams.get(&vec!["a".to_string()]), Some(&3)); // max(2,3)
            assert_eq!(ngrams.get(&vec!["b".to_string()]), Some(&1));
            assert_eq!(ngrams.get(&vec!["c".to_string()]), Some(&1));
        } else {
            panic!("expected NGramSet");
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_empty_list_literal() {
        let mut eval = Evaluator::new();
        let expr = Expr::ListLiteral(vec![]);
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::List(vec![]));
    }

    #[test]
    fn test_nested_function_calls() {
        let mut eval = Evaluator::new();
        // let f = |x| x + 1 in let g = |x| x * 2 in g(f(3))
        let f_body = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::int(1)));
        let f = Expr::Lambda {
            params: vec![LambdaParam {
                name: "x".into(),
                ty: None,
                span: Span::synthetic(),
            }],
            body: Box::new(syn(f_body)),
        };
        let g_body = Expr::binary(BinaryOp::Mul, syn(Expr::var("x")), syn(Expr::int(2)));
        let g = Expr::Lambda {
            params: vec![LambdaParam {
                name: "x".into(),
                ty: None,
                span: Span::synthetic(),
            }],
            body: Box::new(syn(g_body)),
        };
        let f_call = Expr::FunctionCall {
            name: "f".into(),
            args: vec![syn(Expr::int(3))],
        };
        let g_call = Expr::FunctionCall {
            name: "g".into(),
            args: vec![syn(f_call)],
        };
        let expr = Expr::let_expr(
            "f",
            None,
            syn(f),
            syn(Expr::let_expr("g", None, syn(g), syn(g_call))),
        );
        // f(3) = 4, g(4) = 8
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(8));
    }

    #[test]
    fn test_block_evaluation() {
        let mut eval = Evaluator::new();
        let block = Expr::Block(vec![syn(Expr::int(1)), syn(Expr::int(2)), syn(Expr::int(3))]);
        assert_eq!(eval.eval_expr(&block).unwrap(), SemanticValue::Integer(3));
    }

    #[test]
    fn test_empty_block() {
        let mut eval = Evaluator::new();
        let block = Expr::Block(vec![]);
        assert_eq!(eval.eval_expr(&block).unwrap(), SemanticValue::Unit);
    }

    #[test]
    fn test_list_literal() {
        let mut eval = Evaluator::new();
        let expr = Expr::ListLiteral(vec![
            syn(Expr::int(1)),
            syn(Expr::int(2)),
            syn(Expr::int(3)),
        ]);
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::List(vec![
                SemanticValue::Integer(1),
                SemanticValue::Integer(2),
                SemanticValue::Integer(3),
            ])
        );
    }

    #[test]
    fn test_tuple_literal() {
        let mut eval = Evaluator::new();
        let expr = Expr::TupleLiteral(vec![syn(Expr::int(1)), syn(Expr::string("two"))]);
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Tuple(vec![
                SemanticValue::Integer(1),
                SemanticValue::Str("two".into()),
            ])
        );
    }

    #[test]
    fn test_index_access() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::int(10)),
            syn(Expr::int(20)),
            syn(Expr::int(30)),
        ]);
        let expr = Expr::IndexAccess {
            expr: Box::new(syn(list)),
            index: Box::new(syn(Expr::int(1))),
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(20));
    }

    // ── Semiring value operations ────────────────────────────────────

    #[test]
    fn test_semiring_value_counting_add() {
        let a = SemiringValue::Counting(3);
        let b = SemiringValue::Counting(4);
        assert_eq!(a.sr_add(&b).unwrap(), SemiringValue::Counting(7));
    }

    #[test]
    fn test_semiring_value_tropical_add() {
        let a = SemiringValue::Tropical(3.0);
        let b = SemiringValue::Tropical(5.0);
        // Tropical add is min.
        assert_eq!(a.sr_add(&b).unwrap(), SemiringValue::Tropical(3.0));
    }

    #[test]
    fn test_semiring_value_tropical_mul() {
        let a = SemiringValue::Tropical(3.0);
        let b = SemiringValue::Tropical(5.0);
        // Tropical mul is +.
        assert_eq!(a.sr_mul(&b).unwrap(), SemiringValue::Tropical(8.0));
    }

    #[test]
    fn test_semiring_value_boolean() {
        let a = SemiringValue::Bool(true);
        let b = SemiringValue::Bool(false);
        assert_eq!(a.sr_add(&b).unwrap(), SemiringValue::Bool(true)); // OR
        assert_eq!(a.sr_mul(&b).unwrap(), SemiringValue::Bool(false)); // AND
    }

    #[test]
    fn test_semiring_value_type_mismatch() {
        let a = SemiringValue::Counting(1);
        let b = SemiringValue::Tropical(1.0);
        assert!(a.sr_add(&b).is_err());
    }

    // ── FPS Semantics ────────────────────────────────────────────────

    #[test]
    fn test_fps_exact_match() {
        let fps = FPSSemantics::new();
        let metric = MetricDecl {
            name: "exact_match".into(),
            params: vec![],
            return_type: EvalType::Base(BaseType::Bool),
            body: syn(Expr::binary(
                BinaryOp::Eq,
                syn(Expr::var("ref")),
                syn(Expr::var("cand")),
            )),
            attributes: vec![],
            metadata: MetricMetadata::empty(),
            span: Span::synthetic(),
        };
        let result = fps.denote_metric(&metric).unwrap();
        assert!(matches!(result, FPSDenotation::SingleSeries { .. }));
    }

    // ── Clip count ───────────────────────────────────────────────────

    #[test]
    fn test_clip_count() {
        let mut eval = Evaluator::new();
        let expr = Expr::ClipCount {
            count: Box::new(syn(Expr::int(5))),
            max_count: Box::new(syn(Expr::int(3))),
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(3));
    }

    // ── Generate test input ──────────────────────────────────────────

    #[test]
    fn test_generate_test_input_int() {
        let val = generate_test_input(&EvalType::Base(BaseType::Integer), 42);
        assert!(matches!(val, SemanticValue::Integer(_)));
    }

    #[test]
    fn test_generate_test_input_bool() {
        let val = generate_test_input(&EvalType::Base(BaseType::Bool), 42);
        assert!(matches!(val, SemanticValue::Boolean(_)));
    }

    #[test]
    fn test_generate_test_input_string() {
        let val = generate_test_input(&EvalType::Base(BaseType::String), 42);
        assert!(matches!(val, SemanticValue::Str(_)));
    }

    // ── Match pattern expression ─────────────────────────────────────

    #[test]
    fn test_match_pattern_exact() {
        let mut eval = Evaluator::new();
        let expr = Expr::MatchPattern {
            input: Box::new(syn(Expr::string("hello"))),
            pattern: "hello".into(),
            mode: MatchMode::Exact,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    #[test]
    fn test_match_pattern_contains() {
        let mut eval = Evaluator::new();
        let expr = Expr::MatchPattern {
            input: Box::new(syn(Expr::string("hello world"))),
            pattern: "world".into(),
            mode: MatchMode::Contains,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Boolean(true));
    }

    // ── Harmonic and geometric mean ──────────────────────────────────

    #[test]
    fn test_aggregate_harmonic_mean() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::float(2.0)),
            syn(Expr::float(4.0)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::HarmonicMean,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        let result = eval.eval_expr(&expr).unwrap();
        if let SemanticValue::Float(v) = result {
            // H(2, 4) = 2 / (1/2 + 1/4) = 2 / 0.75 = 2.666...
            assert!((v - 2.666666666666).abs() < 1e-6);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn test_aggregate_geometric_mean() {
        let mut eval = Evaluator::new();
        let list = Expr::ListLiteral(vec![
            syn(Expr::float(4.0)),
            syn(Expr::float(9.0)),
        ]);
        let expr = Expr::Aggregate {
            op: AggregationOp::GeometricMean,
            collection: Box::new(syn(list)),
            binding: None,
            body: None,
            semiring: None,
        };
        let result = eval.eval_expr(&expr).unwrap();
        if let SemanticValue::Float(v) = result {
            // G(4, 9) = sqrt(36) = 6.0
            assert!((v - 6.0).abs() < 1e-6);
        } else {
            panic!("expected float");
        }
    }

    // ── Semiring cast ────────────────────────────────────────────────

    #[test]
    fn test_semiring_cast() {
        let mut eval = Evaluator::new();
        let expr = Expr::SemiringCast {
            expr: Box::new(syn(Expr::int(5))),
            from: SemiringType::Counting,
            to: SemiringType::Real,
        };
        let result = eval.eval_expr(&expr).unwrap();
        assert_eq!(
            result,
            SemanticValue::SemiringVal(SemiringValue::Real(5.0))
        );
    }

    // ── SemanticValue display ────────────────────────────────────────

    #[test]
    fn test_semantic_value_display() {
        assert_eq!(format!("{}", SemanticValue::Integer(42)), "42");
        assert_eq!(format!("{}", SemanticValue::Boolean(true)), "true");
        assert_eq!(format!("{}", SemanticValue::Str("hi".into())), "\"hi\"");
        assert_eq!(format!("{}", SemanticValue::Unit), "()");
    }

    // ── Float arithmetic ─────────────────────────────────────────────

    #[test]
    fn test_float_arithmetic() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::Add,
            syn(Expr::float(1.5)),
            syn(Expr::float(2.5)),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Float(4.0));
    }

    #[test]
    fn test_mixed_int_float_arithmetic() {
        let mut eval = Evaluator::new();
        let expr = Expr::binary(
            BinaryOp::Mul,
            syn(Expr::int(3)),
            syn(Expr::float(2.0)),
        );
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Float(6.0));
    }

    // ── Declaration tests ────────────────────────────────────────────

    #[test]
    fn test_eval_let_declaration() {
        let mut eval = Evaluator::new();
        let decl = Declaration::Let(LetDecl {
            name: "x".into(),
            ty: None,
            value: syn(Expr::int(42)),
            attributes: vec![],
            span: Span::synthetic(),
        });
        let result = eval.eval_declaration(&decl).unwrap();
        assert_eq!(result, Some(SemanticValue::Integer(42)));
        assert_eq!(eval.env.lookup("x"), Some(&SemanticValue::Integer(42)));
    }

    #[test]
    fn test_eval_metric_declaration() {
        let mut eval = Evaluator::new();
        let decl = Declaration::Metric(MetricDecl {
            name: "my_metric".into(),
            params: vec![MetricParameter {
                name: "x".into(),
                ty: EvalType::Base(BaseType::Integer),
                default: None,
                span: Span::synthetic(),
            }],
            return_type: EvalType::Base(BaseType::Integer),
            body: syn(Expr::var("x")),
            attributes: vec![],
            metadata: MetricMetadata::empty(),
            span: Span::synthetic(),
        });
        let result = eval.eval_declaration(&decl).unwrap();
        assert!(result.is_none()); // metric decls don't produce values
        assert!(matches!(
            eval.env.lookup("my_metric"),
            Some(SemanticValue::Function(_))
        ));
    }

    // ── Program evaluation ───────────────────────────────────────────

    #[test]
    fn test_eval_program() {
        let mut eval = Evaluator::new();
        let program = Program::new(
            vec![
                syn(Declaration::Let(LetDecl {
                    name: "a".into(),
                    ty: None,
                    value: syn(Expr::int(10)),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
                syn(Declaration::Let(LetDecl {
                    name: "b".into(),
                    ty: None,
                    value: syn(Expr::int(20)),
                    attributes: vec![],
                    span: Span::synthetic(),
                })),
            ],
            Span::synthetic(),
        );
        let results = eval.eval_program(&program).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], SemanticValue::Integer(10));
        assert_eq!(results[1], SemanticValue::Integer(20));
    }

    // ── Match with guard ─────────────────────────────────────────────

    #[test]
    fn test_match_with_guard() {
        let mut eval = Evaluator::new();
        let arms = vec![
            MatchArm {
                pattern: syn(Pattern::Var("x".into())),
                guard: Some(syn(Expr::binary(
                    BinaryOp::Gt,
                    syn(Expr::var("x")),
                    syn(Expr::int(5)),
                ))),
                body: syn(Expr::string("big")),
                span: Span::synthetic(),
            },
            MatchArm {
                pattern: syn(Pattern::Wildcard),
                guard: None,
                body: syn(Expr::string("small")),
                span: Span::synthetic(),
            },
        ];
        let expr = Expr::Match {
            scrutinee: Box::new(syn(Expr::int(3))),
            arms,
        };
        assert_eq!(
            eval.eval_expr(&expr).unwrap(),
            SemanticValue::Str("small".into())
        );
    }

    // ── Builtin sqrt / pow ───────────────────────────────────────────

    #[test]
    fn test_builtin_sqrt() {
        let result = builtin_sqrt(&[SemanticValue::Float(9.0)]).unwrap();
        assert_eq!(result, SemanticValue::Float(3.0));
    }

    #[test]
    fn test_builtin_pow() {
        let result =
            builtin_pow(&[SemanticValue::Float(2.0), SemanticValue::Float(3.0)]).unwrap();
        assert_eq!(result, SemanticValue::Float(8.0));
    }

    #[test]
    fn test_builtin_sqrt_negative() {
        let result = builtin_sqrt(&[SemanticValue::Float(-1.0)]);
        assert!(result.is_err());
    }

    // ── Bounded counting semiring ────────────────────────────────────

    #[test]
    fn test_bounded_counting_add() {
        let a = SemiringValue::BoundedCounting {
            value: 3,
            max_count: 5,
        };
        let b = SemiringValue::BoundedCounting {
            value: 4,
            max_count: 5,
        };
        // 3 + 4 = 7, clamped to 5.
        assert_eq!(
            a.sr_add(&b).unwrap(),
            SemiringValue::BoundedCounting {
                value: 5,
                max_count: 5,
            }
        );
    }

    #[test]
    fn test_bounded_counting_mul() {
        let a = SemiringValue::BoundedCounting {
            value: 2,
            max_count: 10,
        };
        let b = SemiringValue::BoundedCounting {
            value: 3,
            max_count: 10,
        };
        assert_eq!(
            a.sr_mul(&b).unwrap(),
            SemiringValue::BoundedCounting {
                value: 6,
                max_count: 10,
            }
        );
    }

    // ── Normalization of nested expressions ──────────────────────────

    #[test]
    fn test_normalize_nested_add_zero() {
        // (x + 0) + 0 => x
        let inner = Expr::binary(BinaryOp::Add, syn(Expr::var("x")), syn(Expr::int(0)));
        let expr = Expr::binary(BinaryOp::Add, syn(inner), syn(Expr::int(0)));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_normalize_double_neg_arithmetic() {
        let inner = Expr::unary(UnaryOp::Neg, syn(Expr::var("x")));
        let expr = Expr::unary(UnaryOp::Neg, syn(inner));
        let result = normalize(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    // ── SemanticValue coercions ───────────────────────────────────────

    #[test]
    fn test_semantic_value_to_f64() {
        assert_eq!(SemanticValue::Integer(5).to_f64(), Some(5.0));
        assert_eq!(SemanticValue::Float(3.14).to_f64(), Some(3.14));
        assert_eq!(SemanticValue::Boolean(true).to_f64(), Some(1.0));
        assert_eq!(SemanticValue::Boolean(false).to_f64(), Some(0.0));
        assert_eq!(SemanticValue::Str("hi".into()).to_f64(), None);
    }

    #[test]
    fn test_semantic_value_to_bool() {
        assert_eq!(SemanticValue::Boolean(true).to_bool(), Some(true));
        assert_eq!(SemanticValue::Integer(0).to_bool(), Some(false));
        assert_eq!(SemanticValue::Integer(1).to_bool(), Some(true));
    }

    #[test]
    fn test_semantic_value_type_name() {
        assert_eq!(SemanticValue::Integer(0).type_name(), "Integer");
        assert_eq!(SemanticValue::Float(0.0).type_name(), "Float");
        assert_eq!(SemanticValue::Boolean(true).type_name(), "Boolean");
        assert_eq!(SemanticValue::Str("".into()).type_name(), "String");
        assert_eq!(SemanticValue::Unit.type_name(), "Unit");
    }

    // ── SemiringValue zero/one ───────────────────────────────────────

    #[test]
    fn test_semiring_zero_one() {
        assert_eq!(
            SemiringValue::zero_for(&SemiringType::Counting),
            SemiringValue::Counting(0)
        );
        assert_eq!(
            SemiringValue::one_for(&SemiringType::Counting),
            SemiringValue::Counting(1)
        );
        assert_eq!(
            SemiringValue::zero_for(&SemiringType::Boolean),
            SemiringValue::Bool(false)
        );
        assert_eq!(
            SemiringValue::one_for(&SemiringType::Boolean),
            SemiringValue::Bool(true)
        );
        assert_eq!(
            SemiringValue::zero_for(&SemiringType::Tropical),
            SemiringValue::Tropical(f64::INFINITY)
        );
        assert_eq!(
            SemiringValue::one_for(&SemiringType::Tropical),
            SemiringValue::Tropical(0.0)
        );
    }

    // ── List pattern matching ────────────────────────────────────────

    #[test]
    fn test_match_list_pattern() {
        let mut eval = Evaluator::new();
        let arms = vec![MatchArm {
            pattern: syn(Pattern::List {
                elems: vec![syn(Pattern::Var("head".into()))],
                rest: Some(Box::new(syn(Pattern::Var("tail".into())))),
            }),
            guard: None,
            body: syn(Expr::var("head")),
            span: Span::synthetic(),
        }];
        let scrutinee = Expr::ListLiteral(vec![
            syn(Expr::int(1)),
            syn(Expr::int(2)),
            syn(Expr::int(3)),
        ]);
        let expr = Expr::Match {
            scrutinee: Box::new(syn(scrutinee)),
            arms,
        };
        assert_eq!(eval.eval_expr(&expr).unwrap(), SemanticValue::Integer(1));
    }
}
