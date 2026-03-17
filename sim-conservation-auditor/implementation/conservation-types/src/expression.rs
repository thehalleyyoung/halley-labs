//! Symbolic expression representation for conservation analysis.
//!
//! Provides an AST for mathematical expressions that arise from
//! lifting simulation code into symbolic form.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A symbolic variable with optional provenance information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub index: Option<usize>,
    pub is_time_derivative: bool,
    pub derivative_order: u32,
}

impl Variable {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            index: None,
            is_time_derivative: false,
            derivative_order: 0,
        }
    }

    pub fn indexed(name: impl Into<String>, idx: usize) -> Self {
        Self {
            name: name.into(),
            index: Some(idx),
            is_time_derivative: false,
            derivative_order: 0,
        }
    }

    pub fn time_derivative(&self) -> Self {
        Self {
            name: self.name.clone(),
            index: self.index,
            is_time_derivative: true,
            derivative_order: self.derivative_order + 1,
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(idx) = self.index {
            write!(f, "_{}", idx)?;
        }
        for _ in 0..self.derivative_order {
            write!(f, "'")?;
        }
        Ok(())
    }
}

/// Supported unary mathematical operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Ln,
    Log2,
    Log10,
    Sinh,
    Cosh,
    Tanh,
    Floor,
    Ceil,
    Sign,
    Square,
    Cube,
    Reciprocal,
}

impl UnaryOp {
    /// Check if this operation preserves differentiable structure.
    pub fn is_smooth(&self) -> bool {
        matches!(
            self,
            UnaryOp::Neg
                | UnaryOp::Sin
                | UnaryOp::Cos
                | UnaryOp::Tan
                | UnaryOp::Exp
                | UnaryOp::Sinh
                | UnaryOp::Cosh
                | UnaryOp::Tanh
                | UnaryOp::Square
                | UnaryOp::Cube
        )
    }

    /// Get the derivative of this unary operation with respect to its argument.
    pub fn derivative_expr(&self) -> Option<Box<dyn Fn(Expr) -> Expr>> {
        match self {
            UnaryOp::Sin => Some(Box::new(|x| Expr::Unary(UnaryOp::Cos, Box::new(x)))),
            UnaryOp::Cos => Some(Box::new(|x| {
                Expr::Unary(UnaryOp::Neg, Box::new(Expr::Unary(UnaryOp::Sin, Box::new(x))))
            })),
            UnaryOp::Exp => Some(Box::new(|x| Expr::Unary(UnaryOp::Exp, Box::new(x)))),
            UnaryOp::Neg => Some(Box::new(|_| Expr::Constant(-1.0))),
            _ => None,
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Abs => write!(f, "abs"),
            UnaryOp::Sqrt => write!(f, "sqrt"),
            UnaryOp::Sin => write!(f, "sin"),
            UnaryOp::Cos => write!(f, "cos"),
            UnaryOp::Tan => write!(f, "tan"),
            UnaryOp::Asin => write!(f, "asin"),
            UnaryOp::Acos => write!(f, "acos"),
            UnaryOp::Atan => write!(f, "atan"),
            UnaryOp::Exp => write!(f, "exp"),
            UnaryOp::Ln => write!(f, "ln"),
            UnaryOp::Log2 => write!(f, "log2"),
            UnaryOp::Log10 => write!(f, "log10"),
            UnaryOp::Sinh => write!(f, "sinh"),
            UnaryOp::Cosh => write!(f, "cosh"),
            UnaryOp::Tanh => write!(f, "tanh"),
            UnaryOp::Floor => write!(f, "floor"),
            UnaryOp::Ceil => write!(f, "ceil"),
            UnaryOp::Sign => write!(f, "sign"),
            UnaryOp::Square => write!(f, "sq"),
            UnaryOp::Cube => write!(f, "cube"),
            UnaryOp::Reciprocal => write!(f, "1/"),
        }
    }
}

/// Supported binary mathematical operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Min,
    Max,
    Atan2,
    Dot,
    Cross,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "^"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Min => write!(f, "min"),
            BinaryOp::Max => write!(f, "max"),
            BinaryOp::Atan2 => write!(f, "atan2"),
            BinaryOp::Dot => write!(f, "·"),
            BinaryOp::Cross => write!(f, "×"),
        }
    }
}

/// A symbolic mathematical expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// A numeric constant.
    Constant(f64),
    /// A symbolic variable.
    Var(Variable),
    /// A unary operation applied to an expression.
    Unary(UnaryOp, Box<Expr>),
    /// A binary operation on two expressions.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// Summation over a range of indexed expressions.
    Sum {
        index_var: String,
        lower: i64,
        upper: i64,
        body: Box<Expr>,
    },
    /// Product over a range of indexed expressions.
    Product {
        index_var: String,
        lower: i64,
        upper: i64,
        body: Box<Expr>,
    },
    /// Conditional expression.
    IfThenElse {
        condition: Box<BoolExpr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Let binding for intermediate values.
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    /// Function application.
    FnCall {
        name: String,
        args: Vec<Expr>,
    },
    /// Derivative operator d/dt applied to an expression.
    TimeDerivative(Box<Expr>),
    /// Partial derivative ∂/∂x.
    PartialDerivative {
        expr: Box<Expr>,
        var: Variable,
    },
    /// Lie bracket [X, Y].
    LieBracket(Box<Expr>, Box<Expr>),
    /// Poisson bracket {f, g}.
    PoissonBracket(Box<Expr>, Box<Expr>),
    /// A vector of expressions.
    Vector(Vec<Expr>),
    /// Matrix of expressions (row-major).
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<Expr>,
    },
    /// Index into a vector or matrix expression.
    Index(Box<Expr>, Vec<usize>),
}

/// Boolean expression for conditionals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoolExpr {
    True,
    False,
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Not(Box<BoolExpr>),
}

impl Expr {
    /// Create a constant expression.
    pub fn constant(val: f64) -> Self {
        Expr::Constant(val)
    }

    /// Create a variable expression.
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(Variable::new(name))
    }

    /// Create an indexed variable.
    pub fn indexed_var(name: impl Into<String>, idx: usize) -> Self {
        Expr::Var(Variable::indexed(name, idx))
    }

    /// Addition.
    pub fn add(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Add, Box::new(self), Box::new(other))
    }

    /// Subtraction.
    pub fn sub(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Sub, Box::new(self), Box::new(other))
    }

    /// Multiplication.
    pub fn mul(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Mul, Box::new(self), Box::new(other))
    }

    /// Division.
    pub fn div(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Div, Box::new(self), Box::new(other))
    }

    /// Power.
    pub fn pow(self, exponent: Self) -> Self {
        Expr::Binary(BinaryOp::Pow, Box::new(self), Box::new(exponent))
    }

    /// Negation.
    pub fn neg(self) -> Self {
        Expr::Unary(UnaryOp::Neg, Box::new(self))
    }

    /// Sine.
    pub fn sin(self) -> Self {
        Expr::Unary(UnaryOp::Sin, Box::new(self))
    }

    /// Cosine.
    pub fn cos(self) -> Self {
        Expr::Unary(UnaryOp::Cos, Box::new(self))
    }

    /// Exponential.
    pub fn exp(self) -> Self {
        Expr::Unary(UnaryOp::Exp, Box::new(self))
    }

    /// Natural logarithm.
    pub fn ln(self) -> Self {
        Expr::Unary(UnaryOp::Ln, Box::new(self))
    }

    /// Square root.
    pub fn sqrt(self) -> Self {
        Expr::Unary(UnaryOp::Sqrt, Box::new(self))
    }

    /// Collect all free variables in this expression.
    pub fn free_variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort_by(|a, b| a.name.cmp(&b.name));
        vars.dedup_by(|a, b| a.name == b.name && a.index == b.index);
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        match self {
            Expr::Constant(_) => {}
            Expr::Var(v) => vars.push(v.clone()),
            Expr::Unary(_, e) => e.collect_variables(vars),
            Expr::Binary(_, a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Sum { body, .. } | Expr::Product { body, .. } => {
                body.collect_variables(vars);
            }
            Expr::IfThenElse {
                then_expr,
                else_expr,
                ..
            } => {
                then_expr.collect_variables(vars);
                else_expr.collect_variables(vars);
            }
            Expr::Let { value, body, .. } => {
                value.collect_variables(vars);
                body.collect_variables(vars);
            }
            Expr::FnCall { args, .. } => {
                for a in args {
                    a.collect_variables(vars);
                }
            }
            Expr::TimeDerivative(e) => e.collect_variables(vars),
            Expr::PartialDerivative { expr, .. } => expr.collect_variables(vars),
            Expr::LieBracket(a, b) | Expr::PoissonBracket(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Vector(v) => {
                for e in v {
                    e.collect_variables(vars);
                }
            }
            Expr::Matrix { data, .. } => {
                for e in data {
                    e.collect_variables(vars);
                }
            }
            Expr::Index(e, _) => e.collect_variables(vars),
        }
    }

    /// Count the number of nodes in this expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Constant(_) | Expr::Var(_) => 1,
            Expr::Unary(_, e) => 1 + e.node_count(),
            Expr::Binary(_, a, b) => 1 + a.node_count() + b.node_count(),
            Expr::Sum { body, .. } | Expr::Product { body, .. } => 1 + body.node_count(),
            Expr::IfThenElse {
                then_expr,
                else_expr,
                ..
            } => 1 + then_expr.node_count() + else_expr.node_count(),
            Expr::Let { value, body, .. } => 1 + value.node_count() + body.node_count(),
            Expr::FnCall { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
            Expr::TimeDerivative(e) => 1 + e.node_count(),
            Expr::PartialDerivative { expr, .. } => 1 + expr.node_count(),
            Expr::LieBracket(a, b) | Expr::PoissonBracket(a, b) => {
                1 + a.node_count() + b.node_count()
            }
            Expr::Vector(v) => 1 + v.iter().map(|e| e.node_count()).sum::<usize>(),
            Expr::Matrix { data, .. } => 1 + data.iter().map(|e| e.node_count()).sum::<usize>(),
            Expr::Index(e, _) => 1 + e.node_count(),
        }
    }

    /// Maximum depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            Expr::Constant(_) | Expr::Var(_) => 0,
            Expr::Unary(_, e) => 1 + e.depth(),
            Expr::Binary(_, a, b) => 1 + a.depth().max(b.depth()),
            Expr::Sum { body, .. } | Expr::Product { body, .. } => 1 + body.depth(),
            Expr::IfThenElse {
                then_expr,
                else_expr,
                ..
            } => 1 + then_expr.depth().max(else_expr.depth()),
            Expr::Let { value, body, .. } => 1 + value.depth().max(body.depth()),
            Expr::FnCall { args, .. } => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
            Expr::TimeDerivative(e) | Expr::PartialDerivative { expr: e, .. } => 1 + e.depth(),
            Expr::LieBracket(a, b) | Expr::PoissonBracket(a, b) => {
                1 + a.depth().max(b.depth())
            }
            Expr::Vector(v) => 1 + v.iter().map(|e| e.depth()).max().unwrap_or(0),
            Expr::Matrix { data, .. } => {
                1 + data.iter().map(|e| e.depth()).max().unwrap_or(0)
            }
            Expr::Index(e, _) => 1 + e.depth(),
        }
    }

    /// Substitute a variable with an expression.
    pub fn substitute(&self, var_name: &str, replacement: &Expr) -> Expr {
        match self {
            Expr::Constant(c) => Expr::Constant(*c),
            Expr::Var(v) if v.name == var_name => replacement.clone(),
            Expr::Var(v) => Expr::Var(v.clone()),
            Expr::Unary(op, e) => Expr::Unary(*op, Box::new(e.substitute(var_name, replacement))),
            Expr::Binary(op, a, b) => Expr::Binary(
                *op,
                Box::new(a.substitute(var_name, replacement)),
                Box::new(b.substitute(var_name, replacement)),
            ),
            Expr::Vector(v) => {
                Expr::Vector(v.iter().map(|e| e.substitute(var_name, replacement)).collect())
            }
            other => other.clone(),
        }
    }

    /// Check if this expression is a polynomial in the given variables.
    pub fn is_polynomial(&self, vars: &[&str]) -> bool {
        match self {
            Expr::Constant(_) => true,
            Expr::Var(v) => vars.contains(&v.name.as_str()) || !vars.contains(&v.name.as_str()),
            Expr::Unary(UnaryOp::Neg, e) => e.is_polynomial(vars),
            Expr::Binary(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul, a, b) => {
                a.is_polynomial(vars) && b.is_polynomial(vars)
            }
            Expr::Binary(BinaryOp::Pow, base, exp) => {
                if let Expr::Constant(n) = exp.as_ref() {
                    *n >= 0.0 && n.fract() == 0.0 && base.is_polynomial(vars)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Attempt to evaluate the expression with given variable bindings.
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Option<f64> {
        match self {
            Expr::Constant(c) => Some(*c),
            Expr::Var(v) => bindings.get(&v.name).copied(),
            Expr::Unary(op, e) => {
                let val = e.evaluate(bindings)?;
                Some(match op {
                    UnaryOp::Neg => -val,
                    UnaryOp::Abs => val.abs(),
                    UnaryOp::Sqrt => val.sqrt(),
                    UnaryOp::Sin => val.sin(),
                    UnaryOp::Cos => val.cos(),
                    UnaryOp::Tan => val.tan(),
                    UnaryOp::Exp => val.exp(),
                    UnaryOp::Ln => val.ln(),
                    UnaryOp::Square => val * val,
                    UnaryOp::Cube => val * val * val,
                    UnaryOp::Reciprocal => 1.0 / val,
                    UnaryOp::Sinh => val.sinh(),
                    UnaryOp::Cosh => val.cosh(),
                    UnaryOp::Tanh => val.tanh(),
                    UnaryOp::Asin => val.asin(),
                    UnaryOp::Acos => val.acos(),
                    UnaryOp::Atan => val.atan(),
                    UnaryOp::Log2 => val.log2(),
                    UnaryOp::Log10 => val.log10(),
                    UnaryOp::Floor => val.floor(),
                    UnaryOp::Ceil => val.ceil(),
                    UnaryOp::Sign => {
                        if val > 0.0 {
                            1.0
                        } else if val < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                })
            }
            Expr::Binary(op, a, b) => {
                let va = a.evaluate(bindings)?;
                let vb = b.evaluate(bindings)?;
                Some(match op {
                    BinaryOp::Add => va + vb,
                    BinaryOp::Sub => va - vb,
                    BinaryOp::Mul => va * vb,
                    BinaryOp::Div => va / vb,
                    BinaryOp::Pow => va.powf(vb),
                    BinaryOp::Mod => va % vb,
                    BinaryOp::Min => va.min(vb),
                    BinaryOp::Max => va.max(vb),
                    BinaryOp::Atan2 => va.atan2(vb),
                    BinaryOp::Dot | BinaryOp::Cross => return None,
                })
            }
            _ => None,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Constant(c) => write!(f, "{}", c),
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Unary(op, e) => write!(f, "{}({})", op, e),
            Expr::Binary(op, a, b) => write!(f, "({} {} {})", a, op, b),
            Expr::Sum { index_var, lower, upper, body } => {
                write!(f, "Σ_{{{}={}}}^{{{}}} {}", index_var, lower, upper, body)
            }
            Expr::Product { index_var, lower, upper, body } => {
                write!(f, "Π_{{{}={}}}^{{{}}} {}", index_var, lower, upper, body)
            }
            Expr::FnCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
            Expr::TimeDerivative(e) => write!(f, "d/dt({})", e),
            Expr::PartialDerivative { expr, var } => write!(f, "∂/∂{}({})", var, expr),
            Expr::LieBracket(a, b) => write!(f, "[{}, {}]", a, b),
            Expr::PoissonBracket(a, b) => write!(f, "{{{}, {}}}", a, b),
            Expr::Vector(v) => {
                write!(f, "[")?;
                for (i, e) in v.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Expr::Matrix { rows, cols, data } => {
                write!(f, "Mat({}x{})[", rows, cols)?;
                for (i, e) in data.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Expr::Index(e, indices) => {
                write!(f, "{}[", e)?;
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", idx)?;
                }
                write!(f, "]")
            }
            _ => write!(f, "<expr>"),
        }
    }
}

/// Expression simplification engine.
pub struct Simplifier {
    max_iterations: usize,
}

impl Default for Simplifier {
    fn default() -> Self {
        Self { max_iterations: 100 }
    }
}

impl Simplifier {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    /// Simplify an expression by applying algebraic identities.
    pub fn simplify(&self, expr: &Expr) -> Expr {
        let mut current = expr.clone();
        for _ in 0..self.max_iterations {
            let next = self.simplify_step(&current);
            if format!("{:?}", next) == format!("{:?}", current) {
                break;
            }
            current = next;
        }
        current
    }

    fn simplify_step(&self, expr: &Expr) -> Expr {
        match expr {
            // 0 + x = x
            Expr::Binary(BinaryOp::Add, a, b) => {
                let a = self.simplify_step(a);
                let b = self.simplify_step(b);
                match (&a, &b) {
                    (Expr::Constant(0.0), _) => b,
                    (_, Expr::Constant(0.0)) => a,
                    (Expr::Constant(x), Expr::Constant(y)) => Expr::Constant(x + y),
                    _ => Expr::Binary(BinaryOp::Add, Box::new(a), Box::new(b)),
                }
            }
            // x - 0 = x
            Expr::Binary(BinaryOp::Sub, a, b) => {
                let a = self.simplify_step(a);
                let b = self.simplify_step(b);
                match (&a, &b) {
                    (_, Expr::Constant(0.0)) => a,
                    (Expr::Constant(x), Expr::Constant(y)) => Expr::Constant(x - y),
                    _ => Expr::Binary(BinaryOp::Sub, Box::new(a), Box::new(b)),
                }
            }
            // 0 * x = 0, 1 * x = x
            Expr::Binary(BinaryOp::Mul, a, b) => {
                let a = self.simplify_step(a);
                let b = self.simplify_step(b);
                match (&a, &b) {
                    (Expr::Constant(c), _) if *c == 0.0 => Expr::Constant(0.0),
                    (_, Expr::Constant(c)) if *c == 0.0 => Expr::Constant(0.0),
                    (Expr::Constant(c), _) if *c == 1.0 => b,
                    (_, Expr::Constant(c)) if *c == 1.0 => a,
                    (Expr::Constant(x), Expr::Constant(y)) => Expr::Constant(x * y),
                    _ => Expr::Binary(BinaryOp::Mul, Box::new(a), Box::new(b)),
                }
            }
            // x / 1 = x
            Expr::Binary(BinaryOp::Div, a, b) => {
                let a = self.simplify_step(a);
                let b = self.simplify_step(b);
                match (&a, &b) {
                    (_, Expr::Constant(c)) if *c == 1.0 => a,
                    (Expr::Constant(x), Expr::Constant(y)) if *y != 0.0 => Expr::Constant(x / y),
                    _ => Expr::Binary(BinaryOp::Div, Box::new(a), Box::new(b)),
                }
            }
            // x^0 = 1, x^1 = x
            Expr::Binary(BinaryOp::Pow, a, b) => {
                let a = self.simplify_step(a);
                let b = self.simplify_step(b);
                match (&a, &b) {
                    (_, Expr::Constant(c)) if *c == 0.0 => Expr::Constant(1.0),
                    (_, Expr::Constant(c)) if *c == 1.0 => a,
                    _ => Expr::Binary(BinaryOp::Pow, Box::new(a), Box::new(b)),
                }
            }
            // --x = x
            Expr::Unary(UnaryOp::Neg, inner) => {
                let inner = self.simplify_step(inner);
                match &inner {
                    Expr::Unary(UnaryOp::Neg, x) => *x.clone(),
                    Expr::Constant(c) => Expr::Constant(-c),
                    _ => Expr::Unary(UnaryOp::Neg, Box::new(inner)),
                }
            }
            Expr::Unary(op, e) => Expr::Unary(*op, Box::new(self.simplify_step(e))),
            other => other.clone(),
        }
    }
}

/// Builder for constructing expressions from components.
pub struct ExprBuilder {
    counter: usize,
}

impl ExprBuilder {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Create a fresh variable with a unique name.
    pub fn fresh_var(&mut self, prefix: &str) -> Expr {
        let name = format!("{}_{}", prefix, self.counter);
        self.counter += 1;
        Expr::var(name)
    }

    /// Build the kinetic energy expression: T = 0.5 * Σ m_i * |v_i|^2.
    pub fn kinetic_energy(&self, n_particles: usize) -> Expr {
        let mut terms = Vec::new();
        for i in 0..n_particles {
            let m = Expr::indexed_var("m", i);
            let vx = Expr::indexed_var("vx", i);
            let vy = Expr::indexed_var("vy", i);
            let vz = Expr::indexed_var("vz", i);
            let v_squared = vx.clone().mul(vx).add(vy.clone().mul(vy)).add(vz.clone().mul(vz));
            terms.push(Expr::constant(0.5).mul(m).mul(v_squared));
        }
        terms
            .into_iter()
            .reduce(|a, b| a.add(b))
            .unwrap_or(Expr::constant(0.0))
    }

    /// Build gravitational potential energy: V = -G * Σ_{i<j} m_i*m_j / |r_i - r_j|.
    pub fn gravitational_potential(&self, n_particles: usize) -> Expr {
        let g = Expr::var("G");
        let mut terms = Vec::new();
        for i in 0..n_particles {
            for j in (i + 1)..n_particles {
                let mi = Expr::indexed_var("m", i);
                let mj = Expr::indexed_var("m", j);
                let dx = Expr::indexed_var("x", i).sub(Expr::indexed_var("x", j));
                let dy = Expr::indexed_var("y", i).sub(Expr::indexed_var("y", j));
                let dz = Expr::indexed_var("z", i).sub(Expr::indexed_var("z", j));
                let r = dx
                    .clone()
                    .mul(dx)
                    .add(dy.clone().mul(dy))
                    .add(dz.clone().mul(dz))
                    .sqrt();
                terms.push(mi.mul(mj).div(r));
            }
        }
        if terms.is_empty() {
            return Expr::constant(0.0);
        }
        let sum = terms
            .into_iter()
            .reduce(|a, b| a.add(b))
            .unwrap_or(Expr::constant(0.0));
        Expr::Unary(UnaryOp::Neg, Box::new(g.mul(sum)))
    }

    /// Build total linear momentum: P = Σ m_i * v_i.
    pub fn total_momentum(&self, n_particles: usize) -> Expr {
        let mut px_terms = Vec::new();
        let mut py_terms = Vec::new();
        let mut pz_terms = Vec::new();
        for i in 0..n_particles {
            let m = Expr::indexed_var("m", i);
            px_terms.push(m.clone().mul(Expr::indexed_var("vx", i)));
            py_terms.push(m.clone().mul(Expr::indexed_var("vy", i)));
            pz_terms.push(m.mul(Expr::indexed_var("vz", i)));
        }
        let px = px_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        let py = py_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        let pz = pz_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        Expr::Vector(vec![px, py, pz])
    }

    /// Build total angular momentum: L = Σ m_i * (r_i × v_i).
    pub fn total_angular_momentum(&self, n_particles: usize) -> Expr {
        let mut lx_terms = Vec::new();
        let mut ly_terms = Vec::new();
        let mut lz_terms = Vec::new();
        for i in 0..n_particles {
            let m = Expr::indexed_var("m", i);
            let x = Expr::indexed_var("x", i);
            let y = Expr::indexed_var("y", i);
            let z = Expr::indexed_var("z", i);
            let vx = Expr::indexed_var("vx", i);
            let vy = Expr::indexed_var("vy", i);
            let vz = Expr::indexed_var("vz", i);
            lx_terms.push(m.clone().mul(y.clone().mul(vz.clone()).sub(z.clone().mul(vy.clone()))));
            ly_terms.push(m.clone().mul(z.mul(vx.clone()).sub(x.clone().mul(vz))));
            lz_terms.push(m.mul(x.mul(vy).sub(y.mul(vx))));
        }
        let lx = lx_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        let ly = ly_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        let lz = lz_terms.into_iter().reduce(|a, b| a.add(b)).unwrap_or(Expr::constant(0.0));
        Expr::Vector(vec![lx, ly, lz])
    }
}

impl Default for ExprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_display() {
        let v = Variable::new("x");
        assert_eq!(format!("{}", v), "x");
        let vi = Variable::indexed("q", 3);
        assert_eq!(format!("{}", vi), "q_3");
        let vd = v.time_derivative();
        assert_eq!(format!("{}", vd), "x'");
    }

    #[test]
    fn test_expr_construction() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let sum = x.add(y);
        assert_eq!(sum.node_count(), 3);
    }

    #[test]
    fn test_simplify_add_zero() {
        let s = Simplifier::default();
        let e = Expr::constant(0.0).add(Expr::var("x"));
        let result = s.simplify(&e);
        match result {
            Expr::Var(v) => assert_eq!(v.name, "x"),
            other => panic!("Expected Var, got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_mul_one() {
        let s = Simplifier::default();
        let e = Expr::constant(1.0).mul(Expr::var("x"));
        let result = s.simplify(&e);
        match result {
            Expr::Var(v) => assert_eq!(v.name, "x"),
            other => panic!("Expected Var, got {:?}", other),
        }
    }

    #[test]
    fn test_simplify_mul_zero() {
        let s = Simplifier::default();
        let e = Expr::constant(0.0).mul(Expr::var("x"));
        let result = s.simplify(&e);
        match result {
            Expr::Constant(c) => assert_eq!(c, 0.0),
            other => panic!("Expected Constant(0), got {:?}", other),
        }
    }

    #[test]
    fn test_evaluate() {
        let e = Expr::var("x").add(Expr::constant(2.0)).mul(Expr::var("y"));
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 3.0);
        bindings.insert("y".to_string(), 4.0);
        let result = e.evaluate(&bindings).unwrap();
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_variables() {
        let e = Expr::var("x")
            .add(Expr::var("y"))
            .mul(Expr::var("x"));
        let vars = e.free_variables();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_is_polynomial() {
        let e = Expr::var("x").pow(Expr::constant(2.0)).add(Expr::var("y"));
        assert!(e.is_polynomial(&["x", "y"]));
        let non_poly = Expr::var("x").sin();
        assert!(!non_poly.is_polynomial(&["x"]));
    }

    #[test]
    fn test_kinetic_energy_builder() {
        let builder = ExprBuilder::new();
        let ke = builder.kinetic_energy(2);
        assert!(ke.node_count() > 10);
    }

    #[test]
    fn test_gravitational_potential_builder() {
        let builder = ExprBuilder::new();
        let pot = builder.gravitational_potential(3);
        let vars = pot.free_variables();
        assert!(vars.len() > 5);
    }

    #[test]
    fn test_total_momentum_builder() {
        let builder = ExprBuilder::new();
        let mom = builder.total_momentum(2);
        match mom {
            Expr::Vector(v) => assert_eq!(v.len(), 3),
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn test_substitute() {
        let e = Expr::var("x").add(Expr::var("y"));
        let result = e.substitute("x", &Expr::constant(5.0));
        let mut bindings = HashMap::new();
        bindings.insert("y".to_string(), 3.0);
        let val = result.evaluate(&bindings).unwrap();
        assert!((val - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_depth() {
        let e = Expr::var("x")
            .add(Expr::var("y").mul(Expr::var("z")));
        assert_eq!(e.depth(), 2);
    }
}
