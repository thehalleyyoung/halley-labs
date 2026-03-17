//! Expression tree types for floating-point computations.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for expression nodes.
pub type NodeId = Uuid;

/// Generate a new unique node ID.
pub fn new_node_id() -> NodeId {
    Uuid::new_v4()
}

/// Floating-point operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FpOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Sqrt,
    Fma,
    Rem,

    // Transcendental
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Pow,
    Hypot,
    Cbrt,

    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,

    // Comparison/selection
    Min,
    Max,
    Copysign,

    // Rounding
    Floor,
    Ceil,
    Round,
    Trunc,

    // Special
    Erf,
    Erfc,
    Gamma,
    Lgamma,

    // Compound patterns
    SumReduce,
    DotProduct,
    Norm2,
    LogSumExp,

    // Custom operation (extension point)
    Custom(u32),
}

impl FpOp {
    /// Number of operands this operation expects.
    pub fn arity(self) -> usize {
        match self {
            Self::Neg | Self::Abs | Self::Sqrt | Self::Exp | Self::Exp2 | Self::Expm1
            | Self::Log | Self::Log2 | Self::Log10 | Self::Log1p | Self::Cbrt | Self::Sin
            | Self::Cos | Self::Tan | Self::Asin | Self::Acos | Self::Atan | Self::Sinh
            | Self::Cosh | Self::Tanh | Self::Floor | Self::Ceil | Self::Round | Self::Trunc
            | Self::Erf | Self::Erfc | Self::Gamma | Self::Lgamma => 1,

            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem | Self::Pow
            | Self::Hypot | Self::Atan2 | Self::Min | Self::Max | Self::Copysign => 2,

            Self::Fma => 3,

            Self::SumReduce | Self::DotProduct | Self::Norm2 | Self::LogSumExp => 0, // variadic
            Self::Custom(_) => 0,
        }
    }

    /// Whether the operation is variadic.
    pub fn is_variadic(self) -> bool {
        matches!(
            self,
            Self::SumReduce | Self::DotProduct | Self::Norm2 | Self::LogSumExp | Self::Custom(_)
        )
    }

    /// Whether this operation can cause cancellation.
    pub fn can_cancel(self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::SumReduce)
    }

    /// Whether this operation can amplify error.
    pub fn can_amplify(self) -> bool {
        matches!(
            self,
            Self::Div | Self::Pow | Self::Exp | Self::Tan | Self::Atan2
        )
    }

    /// Whether the operation is monotonic in all arguments.
    pub fn is_monotonic(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Neg
                | Self::Sqrt
                | Self::Exp
                | Self::Exp2
                | Self::Expm1
                | Self::Log
                | Self::Log2
                | Self::Log10
                | Self::Log1p
                | Self::Cbrt
                | Self::Sinh
                | Self::Tanh
                | Self::Floor
                | Self::Ceil
                | Self::Round
                | Self::Trunc
        )
    }

    /// Condition number formula: returns the condition number given operand values.
    pub fn condition_number(self, operands: &[f64]) -> f64 {
        match (self, operands) {
            (Self::Add, &[a, b]) => {
                let result = a + b;
                if result.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    (a.abs() + b.abs()) / result.abs()
                }
            }
            (Self::Sub, &[a, b]) => {
                let result = a - b;
                if result.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    (a.abs() + b.abs()) / result.abs()
                }
            }
            (Self::Mul, _) => 1.0,
            (Self::Div, &[_, b]) => {
                if b.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    1.0
                }
            }
            (Self::Sqrt, &[x]) => {
                if x <= 0.0 {
                    f64::INFINITY
                } else {
                    0.5
                }
            }
            (Self::Exp, &[x]) => x.abs(),
            (Self::Log, &[x]) => {
                if x <= 0.0 || (x - 1.0).abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    1.0 / x.ln().abs()
                }
            }
            (Self::Sin, &[x]) => {
                let s = x.sin();
                if s.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    (x * x.cos() / s).abs()
                }
            }
            (Self::Cos, &[x]) => {
                let c = x.cos();
                if c.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    (x * x.sin() / c).abs()
                }
            }
            (Self::Tan, &[x]) => {
                let t = x.tan();
                if t.abs() < f64::MIN_POSITIVE {
                    f64::INFINITY
                } else {
                    (x / (x.sin() * x.cos())).abs()
                }
            }
            (Self::Pow, &[x, y]) => {
                if x <= 0.0 {
                    f64::INFINITY
                } else {
                    y.abs() // condition w.r.t. x; full analysis needs both
                }
            }
            _ => 1.0, // Conservative default
        }
    }

    /// Whether this operation has a well-known numerically stable alternative.
    pub fn has_stable_alternative(self) -> bool {
        matches!(
            self,
            Self::Exp | Self::Log | Self::Sub | Self::SumReduce | Self::DotProduct | Self::Norm2
        )
    }
}

impl fmt::Display for FpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Neg => write!(f, "neg"),
            Self::Abs => write!(f, "abs"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Fma => write!(f, "fma"),
            Self::Rem => write!(f, "rem"),
            Self::Exp => write!(f, "exp"),
            Self::Exp2 => write!(f, "exp2"),
            Self::Expm1 => write!(f, "expm1"),
            Self::Log => write!(f, "log"),
            Self::Log2 => write!(f, "log2"),
            Self::Log10 => write!(f, "log10"),
            Self::Log1p => write!(f, "log1p"),
            Self::Pow => write!(f, "pow"),
            Self::Hypot => write!(f, "hypot"),
            Self::Cbrt => write!(f, "cbrt"),
            Self::Sin => write!(f, "sin"),
            Self::Cos => write!(f, "cos"),
            Self::Tan => write!(f, "tan"),
            Self::Asin => write!(f, "asin"),
            Self::Acos => write!(f, "acos"),
            Self::Atan => write!(f, "atan"),
            Self::Atan2 => write!(f, "atan2"),
            Self::Sinh => write!(f, "sinh"),
            Self::Cosh => write!(f, "cosh"),
            Self::Tanh => write!(f, "tanh"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::Copysign => write!(f, "copysign"),
            Self::Floor => write!(f, "floor"),
            Self::Ceil => write!(f, "ceil"),
            Self::Round => write!(f, "round"),
            Self::Trunc => write!(f, "trunc"),
            Self::Erf => write!(f, "erf"),
            Self::Erfc => write!(f, "erfc"),
            Self::Gamma => write!(f, "gamma"),
            Self::Lgamma => write!(f, "lgamma"),
            Self::SumReduce => write!(f, "sum"),
            Self::DotProduct => write!(f, "dot"),
            Self::Norm2 => write!(f, "norm2"),
            Self::LogSumExp => write!(f, "logsumexp"),
            Self::Custom(id) => write!(f, "custom_{}", id),
        }
    }
}

/// Expression tree node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Literal constant.
    Const(f64),
    /// Named variable.
    Var(String),
    /// Unary operation.
    Unary {
        op: FpOp,
        arg: Box<Expr>,
    },
    /// Binary operation.
    Binary {
        op: FpOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Ternary operation (e.g., FMA).
    Ternary {
        op: FpOp,
        a: Box<Expr>,
        b: Box<Expr>,
        c: Box<Expr>,
    },
    /// Variadic operation (sum, dot product, etc.).
    Variadic {
        op: FpOp,
        args: Vec<Expr>,
    },
    /// Conditional expression.
    IfThenElse {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Let binding.
    Let {
        name: String,
        value: Box<Expr>,
        body: Box<Expr>,
    },
}

impl Expr {
    /// Create a constant expression.
    pub fn constant(value: f64) -> Self {
        Self::Const(value)
    }

    /// Create a variable reference.
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Create an addition.
    pub fn add(left: Expr, right: Expr) -> Self {
        Self::Binary {
            op: FpOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a subtraction.
    pub fn sub(left: Expr, right: Expr) -> Self {
        Self::Binary {
            op: FpOp::Sub,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a multiplication.
    pub fn mul(left: Expr, right: Expr) -> Self {
        Self::Binary {
            op: FpOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a division.
    pub fn div(left: Expr, right: Expr) -> Self {
        Self::Binary {
            op: FpOp::Div,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a unary operation.
    pub fn unary(op: FpOp, arg: Expr) -> Self {
        Self::Unary {
            op,
            arg: Box::new(arg),
        }
    }

    /// Create a binary operation.
    pub fn binary(op: FpOp, left: Expr, right: Expr) -> Self {
        Self::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a FMA(a, b, c) = a*b + c.
    pub fn fma(a: Expr, b: Expr, c: Expr) -> Self {
        Self::Ternary {
            op: FpOp::Fma,
            a: Box::new(a),
            b: Box::new(b),
            c: Box::new(c),
        }
    }

    /// Create a variadic sum.
    pub fn sum(args: Vec<Expr>) -> Self {
        Self::Variadic {
            op: FpOp::SumReduce,
            args,
        }
    }

    /// Create a let binding.
    pub fn let_bind(name: impl Into<String>, value: Expr, body: Expr) -> Self {
        Self::Let {
            name: name.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Count the number of nodes in the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Const(_) | Self::Var(_) => 1,
            Self::Unary { arg, .. } => 1 + arg.node_count(),
            Self::Binary { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Self::Ternary { a, b, c, .. } => {
                1 + a.node_count() + b.node_count() + c.node_count()
            }
            Self::Variadic { args, .. } => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => 1 + cond.node_count() + then_expr.node_count() + else_expr.node_count(),
            Self::Let { value, body, .. } => 1 + value.node_count() + body.node_count(),
        }
    }

    /// Depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Var(_) => 1,
            Self::Unary { arg, .. } => 1 + arg.depth(),
            Self::Binary { left, right, .. } => 1 + left.depth().max(right.depth()),
            Self::Ternary { a, b, c, .. } => {
                1 + a.depth().max(b.depth()).max(c.depth())
            }
            Self::Variadic { args, .. } => {
                1 + args.iter().map(|a| a.depth()).max().unwrap_or(0)
            }
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => {
                1 + cond
                    .depth()
                    .max(then_expr.depth())
                    .max(else_expr.depth())
            }
            Self::Let { value, body, .. } => 1 + value.depth().max(body.depth()),
        }
    }

    /// Collect all variable names used in the expression.
    pub fn free_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, vars: &mut Vec<String>) {
        match self {
            Self::Const(_) => {}
            Self::Var(name) => vars.push(name.clone()),
            Self::Unary { arg, .. } => arg.collect_vars(vars),
            Self::Binary { left, right, .. } => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            Self::Ternary { a, b, c, .. } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
                c.collect_vars(vars);
            }
            Self::Variadic { args, .. } => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => {
                cond.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
            Self::Let { name, value, body } => {
                value.collect_vars(vars);
                let mut body_vars = Vec::new();
                body.collect_vars(&mut body_vars);
                body_vars.retain(|v| v != name);
                vars.extend(body_vars);
            }
        }
    }

    /// Evaluate the expression with the given variable bindings.
    pub fn evaluate(&self, env: &std::collections::HashMap<String, f64>) -> Option<f64> {
        match self {
            Self::Const(v) => Some(*v),
            Self::Var(name) => env.get(name).copied(),
            Self::Unary { op, arg } => {
                let a = arg.evaluate(env)?;
                Some(evaluate_unary(*op, a))
            }
            Self::Binary { op, left, right } => {
                let l = left.evaluate(env)?;
                let r = right.evaluate(env)?;
                Some(evaluate_binary(*op, l, r))
            }
            Self::Ternary { op, a, b, c } => {
                let va = a.evaluate(env)?;
                let vb = b.evaluate(env)?;
                let vc = c.evaluate(env)?;
                Some(evaluate_ternary(*op, va, vb, vc))
            }
            Self::Variadic { op, args } => {
                let vals: Option<Vec<f64>> = args.iter().map(|a| a.evaluate(env)).collect();
                let vals = vals?;
                Some(evaluate_variadic(*op, &vals))
            }
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => {
                let c = cond.evaluate(env)?;
                if c != 0.0 {
                    then_expr.evaluate(env)
                } else {
                    else_expr.evaluate(env)
                }
            }
            Self::Let { name, value, body } => {
                let v = value.evaluate(env)?;
                let mut env2 = env.clone();
                env2.insert(name.clone(), v);
                body.evaluate(&env2)
            }
        }
    }

    /// Substitute a variable with an expression.
    pub fn substitute(&self, var_name: &str, replacement: &Expr) -> Expr {
        match self {
            Self::Const(v) => Self::Const(*v),
            Self::Var(name) => {
                if name == var_name {
                    replacement.clone()
                } else {
                    Self::Var(name.clone())
                }
            }
            Self::Unary { op, arg } => Self::Unary {
                op: *op,
                arg: Box::new(arg.substitute(var_name, replacement)),
            },
            Self::Binary { op, left, right } => Self::Binary {
                op: *op,
                left: Box::new(left.substitute(var_name, replacement)),
                right: Box::new(right.substitute(var_name, replacement)),
            },
            Self::Ternary { op, a, b, c } => Self::Ternary {
                op: *op,
                a: Box::new(a.substitute(var_name, replacement)),
                b: Box::new(b.substitute(var_name, replacement)),
                c: Box::new(c.substitute(var_name, replacement)),
            },
            Self::Variadic { op, args } => Self::Variadic {
                op: *op,
                args: args
                    .iter()
                    .map(|a| a.substitute(var_name, replacement))
                    .collect(),
            },
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => Self::IfThenElse {
                cond: Box::new(cond.substitute(var_name, replacement)),
                then_expr: Box::new(then_expr.substitute(var_name, replacement)),
                else_expr: Box::new(else_expr.substitute(var_name, replacement)),
            },
            Self::Let { name, value, body } => {
                let new_value = value.substitute(var_name, replacement);
                if name == var_name {
                    Self::Let {
                        name: name.clone(),
                        value: Box::new(new_value),
                        body: body.clone(),
                    }
                } else {
                    Self::Let {
                        name: name.clone(),
                        value: Box::new(new_value),
                        body: Box::new(body.substitute(var_name, replacement)),
                    }
                }
            }
        }
    }

    /// Check if the expression is a pure constant (no variables).
    pub fn is_constant(&self) -> bool {
        self.free_variables().is_empty()
    }

    /// Map a function over all nodes bottom-up.
    pub fn map<F: Fn(&Expr) -> Expr>(&self, f: &F) -> Expr {
        let mapped = match self {
            Self::Const(_) | Self::Var(_) => self.clone(),
            Self::Unary { op, arg } => Self::Unary {
                op: *op,
                arg: Box::new(arg.map(f)),
            },
            Self::Binary { op, left, right } => Self::Binary {
                op: *op,
                left: Box::new(left.map(f)),
                right: Box::new(right.map(f)),
            },
            Self::Ternary { op, a, b, c } => Self::Ternary {
                op: *op,
                a: Box::new(a.map(f)),
                b: Box::new(b.map(f)),
                c: Box::new(c.map(f)),
            },
            Self::Variadic { op, args } => Self::Variadic {
                op: *op,
                args: args.iter().map(|a| a.map(f)).collect(),
            },
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => Self::IfThenElse {
                cond: Box::new(cond.map(f)),
                then_expr: Box::new(then_expr.map(f)),
                else_expr: Box::new(else_expr.map(f)),
            },
            Self::Let { name, value, body } => Self::Let {
                name: name.clone(),
                value: Box::new(value.map(f)),
                body: Box::new(body.map(f)),
            },
        };
        f(&mapped)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => write!(f, "{}", v),
            Self::Var(name) => write!(f, "{}", name),
            Self::Unary { op, arg } => write!(f, "{}({})", op, arg),
            Self::Binary { op, left, right } => write!(f, "({} {} {})", left, op, right),
            Self::Ternary { op, a, b, c } => write!(f, "{}({}, {}, {})", op, a, b, c),
            Self::Variadic { op, args } => {
                write!(f, "{}(", op)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Self::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => write!(f, "if {} then {} else {}", cond, then_expr, else_expr),
            Self::Let { name, value, body } => {
                write!(f, "let {} = {} in {}", name, value, body)
            }
        }
    }
}

/// Evaluate a unary operation.
pub fn evaluate_unary(op: FpOp, a: f64) -> f64 {
    match op {
        FpOp::Neg => -a,
        FpOp::Abs => a.abs(),
        FpOp::Sqrt => a.sqrt(),
        FpOp::Exp => a.exp(),
        FpOp::Exp2 => a.exp2(),
        FpOp::Expm1 => a.exp_m1(),
        FpOp::Log => a.ln(),
        FpOp::Log2 => a.log2(),
        FpOp::Log10 => a.log10(),
        FpOp::Log1p => a.ln_1p(),
        FpOp::Cbrt => a.cbrt(),
        FpOp::Sin => a.sin(),
        FpOp::Cos => a.cos(),
        FpOp::Tan => a.tan(),
        FpOp::Asin => a.asin(),
        FpOp::Acos => a.acos(),
        FpOp::Atan => a.atan(),
        FpOp::Sinh => a.sinh(),
        FpOp::Cosh => a.cosh(),
        FpOp::Tanh => a.tanh(),
        FpOp::Floor => a.floor(),
        FpOp::Ceil => a.ceil(),
        FpOp::Round => a.round(),
        FpOp::Trunc => a.trunc(),
        _ => f64::NAN,
    }
}

/// Evaluate a binary operation.
pub fn evaluate_binary(op: FpOp, a: f64, b: f64) -> f64 {
    match op {
        FpOp::Add => a + b,
        FpOp::Sub => a - b,
        FpOp::Mul => a * b,
        FpOp::Div => a / b,
        FpOp::Rem => a % b,
        FpOp::Pow => a.powf(b),
        FpOp::Hypot => a.hypot(b),
        FpOp::Atan2 => a.atan2(b),
        FpOp::Min => a.min(b),
        FpOp::Max => a.max(b),
        FpOp::Copysign => a.copysign(b),
        _ => f64::NAN,
    }
}

/// Evaluate a ternary operation.
pub fn evaluate_ternary(op: FpOp, a: f64, b: f64, c: f64) -> f64 {
    match op {
        FpOp::Fma => a.mul_add(b, c),
        _ => f64::NAN,
    }
}

/// Evaluate a variadic operation.
pub fn evaluate_variadic(op: FpOp, args: &[f64]) -> f64 {
    match op {
        FpOp::SumReduce => args.iter().sum(),
        FpOp::DotProduct => {
            // Expects pairs: [a0, b0, a1, b1, ...]
            args.chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        pair[0] * pair[1]
                    } else {
                        0.0
                    }
                })
                .sum()
        }
        FpOp::Norm2 => args.iter().map(|x| x * x).sum::<f64>().sqrt(),
        FpOp::LogSumExp => {
            if args.is_empty() {
                return f64::NEG_INFINITY;
            }
            let max = args.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max.is_infinite() {
                return max;
            }
            max + args.iter().map(|&x| (x - max).exp()).sum::<f64>().ln()
        }
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_basic_expression() {
        let expr = Expr::add(Expr::var("x"), Expr::constant(1.0));
        let mut env = HashMap::new();
        env.insert("x".to_string(), 2.0);
        assert_eq!(expr.evaluate(&env), Some(3.0));
    }

    #[test]
    fn test_expression_tree_properties() {
        let expr = Expr::mul(
            Expr::add(Expr::var("x"), Expr::var("y")),
            Expr::sub(Expr::var("x"), Expr::var("y")),
        );
        assert_eq!(expr.node_count(), 5);
        assert_eq!(expr.depth(), 3);
        assert_eq!(expr.free_variables(), vec!["x", "y"]);
    }

    #[test]
    fn test_substitution() {
        let expr = Expr::add(Expr::var("x"), Expr::constant(1.0));
        let substituted = expr.substitute("x", &Expr::constant(42.0));
        let env = HashMap::new();
        assert_eq!(substituted.evaluate(&env), Some(43.0));
    }

    #[test]
    fn test_let_binding() {
        let expr = Expr::let_bind(
            "t",
            Expr::add(Expr::var("x"), Expr::constant(1.0)),
            Expr::mul(Expr::var("t"), Expr::var("t")),
        );
        let mut env = HashMap::new();
        env.insert("x".to_string(), 2.0);
        assert_eq!(expr.evaluate(&env), Some(9.0));
    }

    #[test]
    fn test_variadic_sum() {
        let expr = Expr::sum(vec![
            Expr::constant(1.0),
            Expr::constant(2.0),
            Expr::constant(3.0),
        ]);
        let env = HashMap::new();
        assert_eq!(expr.evaluate(&env), Some(6.0));
    }

    #[test]
    fn test_logsumexp() {
        let args = vec![1.0, 2.0, 3.0];
        let result = evaluate_variadic(FpOp::LogSumExp, &args);
        // log(e^1 + e^2 + e^3) ≈ 3.4076
        assert!((result - 3.4076).abs() < 0.001);
    }

    #[test]
    fn test_condition_numbers() {
        // Subtraction of nearly equal numbers: very high condition number
        let cn = FpOp::Sub.condition_number(&[1.0, 1.0 - 1e-15]);
        assert!(cn > 1e14);

        // Multiplication: condition number is 1
        let cn_mul = FpOp::Mul.condition_number(&[100.0, 200.0]);
        assert_eq!(cn_mul, 1.0);

        // Square root: condition number is 0.5
        let cn_sqrt = FpOp::Sqrt.condition_number(&[4.0]);
        assert_eq!(cn_sqrt, 0.5);
    }

    #[test]
    fn test_op_properties() {
        assert!(FpOp::Sub.can_cancel());
        assert!(FpOp::Exp.can_amplify());
        assert!(FpOp::Add.is_monotonic());
        assert_eq!(FpOp::Add.arity(), 2);
        assert_eq!(FpOp::Sqrt.arity(), 1);
        assert!(FpOp::SumReduce.is_variadic());
    }

    #[test]
    fn test_display() {
        let expr = Expr::add(Expr::var("x"), Expr::constant(1.0));
        assert_eq!(format!("{}", expr), "(x + 1)");
    }
}
