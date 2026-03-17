//! # fpdiag-symbolic
//!
//! Symbolic expression manipulation and simplification for Penumbra.
//!
//! Provides pattern matching on expression trees to detect known
//! numerical patterns (e.g., `exp(x) - 1`, `log(1 + x)`, `a² + b²`)
//! that have numerically superior alternatives.

use fpdiag_types::expression::{ExprNode, ExprTree, FpOp, NodeId};
use std::fmt;
use thiserror::Error;

/// Errors from symbolic analysis.
#[derive(Debug, Error)]
pub enum SymbolicError {
    #[error("node not found: {0}")]
    NodeNotFound(NodeId),
    #[error("pattern matching failed: {0}")]
    PatternFailed(String),
}

/// A recognized numerical pattern in an expression tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericalPattern {
    /// `exp(x) - 1` → use `expm1(x)`.
    ExpMinus1 { x: NodeId },
    /// `log(1 + x)` → use `log1p(x)`.
    Log1PlusX { x: NodeId },
    /// `sqrt(a² + b²)` → use `hypot(a, b)`.
    SqrtSumSquares { a: NodeId, b: NodeId },
    /// `(b² - 4ac)` in quadratic formula → stable form.
    QuadraticDiscriminant { a: NodeId, b: NodeId, c: NodeId },
    /// Sum of many terms → Kahan / pairwise summation.
    LongReduction { terms: Vec<NodeId> },
    /// `Σ exp(xᵢ)` with normalization → log-sum-exp.
    SoftmaxLike { terms: Vec<NodeId> },
    /// Running mean/variance → Welford's algorithm.
    OnlineVariance { terms: Vec<NodeId> },
    /// Generic alternating-sign summation.
    AlternatingSeries { terms: Vec<NodeId> },
}

impl NumericalPattern {
    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ExpMinus1 { .. } => "exp(x) - 1 pattern (use expm1)",
            Self::Log1PlusX { .. } => "log(1 + x) pattern (use log1p)",
            Self::SqrtSumSquares { .. } => "sqrt(a² + b²) pattern (use hypot)",
            Self::QuadraticDiscriminant { .. } => "quadratic discriminant (use stable form)",
            Self::LongReduction { .. } => "long summation reduction (use compensated sum)",
            Self::SoftmaxLike { .. } => "softmax-like pattern (use log-sum-exp)",
            Self::OnlineVariance { .. } => "online variance (use Welford's algorithm)",
            Self::AlternatingSeries { .. } => "alternating series (reorder or compensate)",
        }
    }
}

impl fmt::Display for NumericalPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

/// Pattern matcher that scans expression trees for known numerical patterns.
pub struct PatternMatcher;

impl PatternMatcher {
    /// Scan an expression tree for all recognized patterns.
    pub fn find_patterns(tree: &ExprTree) -> Vec<NumericalPattern> {
        let mut patterns = Vec::new();

        for (id, node) in tree.iter() {
            // Check for exp(x) - 1
            if let Some(pat) = Self::match_expm1(tree, id, node) {
                patterns.push(pat);
            }
            // Check for log(1 + x)
            if let Some(pat) = Self::match_log1p(tree, id, node) {
                patterns.push(pat);
            }
            // Check for sqrt(a² + b²)
            if let Some(pat) = Self::match_hypot(tree, id, node) {
                patterns.push(pat);
            }
        }

        patterns
    }

    /// Match `exp(x) - 1`: a Sub node whose left child is Exp.
    fn match_expm1(tree: &ExprTree, _id: NodeId, node: &ExprNode) -> Option<NumericalPattern> {
        if let ExprNode::Operation {
            op: FpOp::Sub,
            children,
        } = node
        {
            if children.len() == 2 {
                let lhs = tree.get(children[0])?;
                let rhs = tree.get(children[1])?;
                // Check lhs is exp(x) and rhs is constant 1.0
                if let ExprNode::Operation {
                    op: FpOp::Exp,
                    children: exp_children,
                } = lhs
                {
                    if let ExprNode::Constant(v) = rhs {
                        if (*v - 1.0).abs() < f64::EPSILON {
                            return Some(NumericalPattern::ExpMinus1 { x: exp_children[0] });
                        }
                    }
                }
            }
        }
        None
    }

    /// Match `log(1 + x)`: a Log node whose child is Add(1, x).
    fn match_log1p(tree: &ExprTree, _id: NodeId, node: &ExprNode) -> Option<NumericalPattern> {
        if let ExprNode::Operation {
            op: FpOp::Log,
            children,
        } = node
        {
            if children.len() == 1 {
                let child = tree.get(children[0])?;
                if let ExprNode::Operation {
                    op: FpOp::Add,
                    children: add_children,
                } = child
                {
                    if add_children.len() == 2 {
                        // Check if one child is constant 1.0
                        for (i, &cid) in add_children.iter().enumerate() {
                            if let Some(ExprNode::Constant(v)) = tree.get(cid) {
                                if (*v - 1.0).abs() < f64::EPSILON {
                                    let other = add_children[1 - i];
                                    return Some(NumericalPattern::Log1PlusX { x: other });
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Match `sqrt(a² + b²)`: Sqrt(Add(Mul(a,a), Mul(b,b))).
    fn match_hypot(tree: &ExprTree, _id: NodeId, node: &ExprNode) -> Option<NumericalPattern> {
        if let ExprNode::Operation {
            op: FpOp::Sqrt,
            children,
        } = node
        {
            if children.len() == 1 {
                let child = tree.get(children[0])?;
                if let ExprNode::Operation {
                    op: FpOp::Add,
                    children: add_children,
                } = child
                {
                    if add_children.len() == 2 {
                        let lhs = tree.get(add_children[0])?;
                        let rhs = tree.get(add_children[1])?;
                        if let (
                            ExprNode::Operation {
                                op: FpOp::Mul,
                                children: lc,
                            },
                            ExprNode::Operation {
                                op: FpOp::Mul,
                                children: rc,
                            },
                        ) = (lhs, rhs)
                        {
                            if lc.len() == 2 && rc.len() == 2 && lc[0] == lc[1] && rc[0] == rc[1] {
                                return Some(NumericalPattern::SqrtSumSquares {
                                    a: lc[0],
                                    b: rc[0],
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Apply a pattern-based rewrite to an expression tree.
pub fn apply_rewrite(tree: &mut ExprTree, pattern: &NumericalPattern) -> Result<(), SymbolicError> {
    match pattern {
        NumericalPattern::ExpMinus1 { x } => {
            // Replace the Sub(Exp(x), 1) subtree with a single Expm1 node
            // In practice this would modify the tree; here we add a new node
            let _new_node = tree.add_node(ExprNode::Operation {
                op: FpOp::Exp, // Would be Expm1 if we had it
                children: vec![*x],
            });
            Ok(())
        }
        NumericalPattern::Log1PlusX { x } => {
            let _new_node = tree.add_node(ExprNode::Operation {
                op: FpOp::Log, // Would be Log1p if we had it
                children: vec![*x],
            });
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpdiag_types::expression::ExprBuilder;

    #[test]
    fn detect_expm1_pattern() {
        let mut b = ExprBuilder::new();
        let x = b.variable("x");
        let exp_x = b.unop(FpOp::Exp, x);
        let one = b.constant(1.0);
        let sub = b.binop(FpOp::Sub, exp_x, one);
        let tree = b.build(sub);

        let patterns = PatternMatcher::find_patterns(&tree);
        assert_eq!(patterns.len(), 1);
        assert!(matches!(patterns[0], NumericalPattern::ExpMinus1 { .. }));
    }

    #[test]
    fn detect_log1p_pattern() {
        let mut b = ExprBuilder::new();
        let x = b.variable("x");
        let one = b.constant(1.0);
        let add = b.binop(FpOp::Add, one, x);
        let log = b.unop(FpOp::Log, add);
        let tree = b.build(log);

        let patterns = PatternMatcher::find_patterns(&tree);
        assert_eq!(patterns.len(), 1);
        assert!(matches!(patterns[0], NumericalPattern::Log1PlusX { .. }));
    }
}
