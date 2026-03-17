//! Expression rewriting and normalization for Penumbra.

use penumbra_types::FpOperation;
use serde::{Deserialize, Serialize};

/// An expression tree representing floating-point computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Const(f64),
    Var(String),
    UnaryOp {
        op: FpOperation,
        arg: Box<Expr>,
    },
    BinaryOp {
        op: FpOperation,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    TernaryOp {
        op: FpOperation,
        a: Box<Expr>,
        b: Box<Expr>,
        c: Box<Expr>,
    },
}

impl Expr {
    pub fn constant(v: f64) -> Self {
        Expr::Const(v)
    }

    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    pub fn unary(op: FpOperation, arg: Expr) -> Self {
        Expr::UnaryOp { op, arg: Box::new(arg) }
    }

    pub fn binary(op: FpOperation, left: Expr, right: Expr) -> Self {
        Expr::BinaryOp { op, left: Box::new(left), right: Box::new(right) }
    }

    pub fn depth(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) => 0,
            Expr::UnaryOp { arg, .. } => 1 + arg.depth(),
            Expr::BinaryOp { left, right, .. } => 1 + left.depth().max(right.depth()),
            Expr::TernaryOp { a, b, c, .. } => {
                1 + a.depth().max(b.depth()).max(c.depth())
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) => 1,
            Expr::UnaryOp { arg, .. } => 1 + arg.node_count(),
            Expr::BinaryOp { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }
            Expr::TernaryOp { a, b, c, .. } => {
                1 + a.node_count() + b.node_count() + c.node_count()
            }
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(v) => write!(f, "{v}"),
            Expr::Var(name) => write!(f, "{name}"),
            Expr::UnaryOp { op, arg } => write!(f, "{op}({arg})"),
            Expr::BinaryOp { op, left, right } => {
                write!(f, "({left} {op} {right})")
            }
            Expr::TernaryOp { op, a, b, c } => {
                write!(f, "{op}({a}, {b}, {c})")
            }
        }
    }
}
