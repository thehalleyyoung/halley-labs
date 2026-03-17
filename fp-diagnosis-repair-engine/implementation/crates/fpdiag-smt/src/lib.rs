//! # fpdiag-smt
//!
//! SMT solver integration for Penumbra repair validation.
//!
//! Provides an interface for encoding floating-point repair correctness
//! queries into SMT-LIB format and dispatching them to Z3 or other
//! FP-theory-capable solvers.

use fpdiag_types::expression::{ExprNode, ExprTree, FpOp, NodeId};
use std::fmt;
use thiserror::Error;

/// Errors from the SMT module.
#[derive(Debug, Error)]
pub enum SmtError {
    #[error("solver not found: {0}")]
    SolverNotFound(String),
    #[error("encoding error: {0}")]
    EncodingError(String),
    #[error("solver timeout after {0}ms")]
    Timeout(u64),
    #[error("solver returned unknown")]
    Unknown,
}

/// Result of an SMT query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtResult {
    Sat,
    Unsat,
    Unknown,
}

impl fmt::Display for SmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sat => write!(f, "sat"),
            Self::Unsat => write!(f, "unsat"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// An SMT-LIB formula for floating-point verification.
#[derive(Debug, Clone)]
pub struct SmtFormula {
    /// The SMT-LIB2 string representation.
    pub smt_lib: String,
    /// Human-readable description.
    pub description: String,
}

/// Encoder that translates expression trees into SMT-LIB format.
pub struct SmtEncoder {
    timeout_ms: u64,
}

impl SmtEncoder {
    /// Create a new encoder with the given solver timeout.
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }

    /// Encode the assertion that `original_error > repaired_error`
    /// for all inputs in the domain.
    pub fn encode_error_reduction(
        &self,
        original: &ExprTree,
        repaired: &ExprTree,
    ) -> Result<SmtFormula, SmtError> {
        let mut smt = String::new();
        smt.push_str("(set-logic QF_FP)\n");
        smt.push_str(&format!("(set-option :timeout {})\n", self.timeout_ms));
        smt.push_str("\n; Original expression\n");

        // Encode variables
        let vars = self.collect_variables(original);
        for var in &vars {
            smt.push_str(&format!("(declare-const {} Float64)\n", var));
        }

        // Encode original expression
        if let Some(root) = original.root {
            let orig_expr = self.encode_expr(original, root)?;
            smt.push_str(&format!("(define-fun original () Float64 {})\n", orig_expr));
        }

        // Encode repaired expression
        if let Some(root) = repaired.root {
            let rep_expr = self.encode_expr(repaired, root)?;
            smt.push_str(&format!("(define-fun repaired () Float64 {})\n", rep_expr));
        }

        // Assert that repaired error is NOT smaller (looking for counterexample)
        smt.push_str("\n; Assert repaired error >= original error (negation)\n");
        smt.push_str("(assert (fp.geq (fp.abs (fp.sub RNE repaired original)) ");
        smt.push_str("(fp.abs (fp.sub RNE original original))))\n");
        smt.push_str("(check-sat)\n");

        Ok(SmtFormula {
            smt_lib: smt,
            description: "Error reduction verification".to_string(),
        })
    }

    /// Collect variable names from an expression tree.
    fn collect_variables(&self, tree: &ExprTree) -> Vec<String> {
        let mut vars = Vec::new();
        for (_id, node) in tree.iter() {
            if let ExprNode::Variable { name, .. } = node {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
        }
        vars
    }

    /// Encode a single expression node into SMT-LIB.
    fn encode_expr(&self, tree: &ExprTree, id: NodeId) -> Result<String, SmtError> {
        let node = tree
            .get(id)
            .ok_or_else(|| SmtError::EncodingError(format!("node {} not found", id)))?;

        match node {
            ExprNode::Constant(v) => Ok(format!(
                "(fp #b{} #b{:011b} #b{:052b})",
                if *v < 0.0 { "1" } else { "0" },
                ((v.to_bits() >> 52) & 0x7FF),
                (v.to_bits() & 0xFFFFFFFFFFFFF)
            )),
            ExprNode::Variable { name, .. } => Ok(name.clone()),
            ExprNode::Operation { op, children } => {
                let child_exprs: Result<Vec<String>, SmtError> = children
                    .iter()
                    .map(|&c| self.encode_expr(tree, c))
                    .collect();
                let child_exprs = child_exprs?;

                let smt_op = match op {
                    FpOp::Add => "fp.add RNE",
                    FpOp::Sub => "fp.sub RNE",
                    FpOp::Mul => "fp.mul RNE",
                    FpOp::Div => "fp.div RNE",
                    FpOp::Neg => "fp.neg",
                    FpOp::Abs => "fp.abs",
                    FpOp::Sqrt => "fp.sqrt RNE",
                    FpOp::Fma => "fp.fma RNE",
                    FpOp::Min => "fp.min",
                    FpOp::Max => "fp.max",
                    _ => {
                        return Err(SmtError::EncodingError(format!(
                            "unsupported operation for SMT encoding: {}",
                            op
                        )))
                    }
                };

                Ok(format!("({} {})", smt_op, child_exprs.join(" ")))
            }
            ExprNode::LibraryCall { function, .. } => Err(SmtError::EncodingError(format!(
                "cannot encode library call '{}' in SMT",
                function
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::expression::ExprBuilder;

    #[test]
    fn encode_simple_addition() {
        let mut b = ExprBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.binop(FpOp::Add, x, y);
        let tree = b.build(sum);

        let encoder = SmtEncoder::new(5000);
        let vars = encoder.collect_variables(&tree);
        assert_eq!(vars.len(), 2);
    }
}
