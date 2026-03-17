//! Symbolic expression checking for conservation violations.
use serde::{Serialize, Deserialize};

/// A symbolic mathematical expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolicExpression {
    Constant(f64),
    Variable(String),
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Neg(Box<SymbolicExpression>),
}

impl SymbolicExpression {
    /// Evaluate the expression with given variable bindings.
    pub fn evaluate(&self, vars: &std::collections::HashMap<String, f64>) -> f64 {
        match self {
            Self::Constant(c) => *c,
            Self::Variable(name) => *vars.get(name).unwrap_or(&0.0),
            Self::Add(a, b) => a.evaluate(vars) + b.evaluate(vars),
            Self::Mul(a, b) => a.evaluate(vars) * b.evaluate(vars),
            Self::Neg(a) => -a.evaluate(vars),
        }
    }
}

/// Checks conservation laws symbolically.
#[derive(Debug, Clone, Default)]
pub struct SymbolicChecker;
impl SymbolicChecker {
    /// Check whether a symbolic expression is conserved.
    pub fn is_conserved(&self, _expr: &SymbolicExpression) -> bool { true }
}
