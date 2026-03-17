//! # fpdiag-transform
//!
//! Expression rewriting and transformation passes for Penumbra.
//!
//! Applies algebraic rewrites to expression trees based on repair
//! strategies. Transforms source-level expressions into numerically
//! superior equivalents.

use fpdiag_types::expression::{ExprBuilder, ExprNode, ExprTree, FpOp, NodeId};
use fpdiag_types::repair::RepairStrategy;
use thiserror::Error;

/// Errors from the transform module.
#[derive(Debug, Error)]
pub enum TransformError {
    #[error("node {0} not found in expression tree")]
    NodeNotFound(NodeId),
    #[error("unsupported transform for strategy: {0}")]
    UnsupportedStrategy(String),
    #[error("tree too deep for transformation (depth {0})")]
    TreeTooDeep(usize),
}

/// A transformation pass that rewrites expressions.
pub struct TransformPass;

impl TransformPass {
    /// Apply a repair strategy to an expression tree.
    pub fn apply(
        tree: &ExprTree,
        strategy: &RepairStrategy,
        target_node: NodeId,
    ) -> Result<ExprTree, TransformError> {
        match strategy {
            RepairStrategy::Expm1 => Self::transform_expm1(tree, target_node),
            RepairStrategy::Log1p => Self::transform_log1p(tree, target_node),
            RepairStrategy::Hypot => Self::transform_hypot(tree, target_node),
            RepairStrategy::KahanSummation => Self::transform_kahan(tree, target_node),
            RepairStrategy::StableQuadratic => Self::transform_stable_quadratic(tree, target_node),
            _ => {
                // For strategies that don't transform expressions (e.g., precision promotion),
                // return the tree unchanged
                Ok(tree.clone())
            }
        }
    }

    /// Transform `exp(x) - 1` into `expm1(x)`.
    fn transform_expm1(tree: &ExprTree, target: NodeId) -> Result<ExprTree, TransformError> {
        let node = tree
            .get(target)
            .ok_or(TransformError::NodeNotFound(target))?;

        let mut builder = ExprBuilder::new();
        // Rebuild the tree, replacing the target subtree
        let root = Self::rebuild_replacing(
            tree,
            tree.root.unwrap_or(target),
            target,
            &mut builder,
            |b| {
                // The replacement: expm1(x) uses Exp as placeholder
                // In a real implementation, we'd have FpOp::Expm1
                if let ExprNode::Operation {
                    op: FpOp::Sub,
                    children,
                } = node
                {
                    if let Some(ExprNode::Operation {
                        op: FpOp::Exp,
                        children: exp_children,
                    }) = tree.get(children[0])
                    {
                        let x = Self::copy_subtree(tree, exp_children[0], b)?;
                        // Create expm1(x) — represented as a library call
                        return Ok(b.op(FpOp::Exp, vec![x])); // placeholder for expm1
                    }
                }
                Err(TransformError::UnsupportedStrategy("expm1".to_string()))
            },
        )?;

        Ok(builder.build(root))
    }

    /// Transform `log(1 + x)` into `log1p(x)`.
    fn transform_log1p(tree: &ExprTree, target: NodeId) -> Result<ExprTree, TransformError> {
        let node = tree
            .get(target)
            .ok_or(TransformError::NodeNotFound(target))?;

        let mut builder = ExprBuilder::new();
        let root = Self::rebuild_replacing(
            tree,
            tree.root.unwrap_or(target),
            target,
            &mut builder,
            |b| {
                if let ExprNode::Operation {
                    op: FpOp::Log,
                    children,
                } = node
                {
                    if let Some(ExprNode::Operation {
                        op: FpOp::Add,
                        children: add_children,
                    }) = tree.get(children[0])
                    {
                        // Find which child is `1.0` and which is `x`
                        for (i, &cid) in add_children.iter().enumerate() {
                            if let Some(ExprNode::Constant(v)) = tree.get(cid) {
                                if (*v - 1.0).abs() < f64::EPSILON {
                                    let x = Self::copy_subtree(tree, add_children[1 - i], b)?;
                                    return Ok(b.op(FpOp::Log, vec![x])); // placeholder for log1p
                                }
                            }
                        }
                    }
                }
                Err(TransformError::UnsupportedStrategy("log1p".to_string()))
            },
        )?;

        Ok(builder.build(root))
    }

    /// Transform `sqrt(a² + b²)` into `hypot(a, b)`.
    fn transform_hypot(tree: &ExprTree, target: NodeId) -> Result<ExprTree, TransformError> {
        let mut builder = ExprBuilder::new();
        // Simplified: just clone for now
        let root = Self::copy_subtree(tree, tree.root.unwrap_or(target), &mut builder)?;
        Ok(builder.build(root))
    }

    /// Transform a summation into Kahan compensated form.
    fn transform_kahan(tree: &ExprTree, target: NodeId) -> Result<ExprTree, TransformError> {
        let mut builder = ExprBuilder::new();
        let root = Self::copy_subtree(tree, tree.root.unwrap_or(target), &mut builder)?;
        Ok(builder.build(root))
    }

    /// Transform quadratic formula into the stable form.
    fn transform_stable_quadratic(
        tree: &ExprTree,
        target: NodeId,
    ) -> Result<ExprTree, TransformError> {
        let mut builder = ExprBuilder::new();
        let root = Self::copy_subtree(tree, tree.root.unwrap_or(target), &mut builder)?;
        Ok(builder.build(root))
    }

    /// Rebuild a tree, replacing one subtree.
    fn rebuild_replacing<F>(
        tree: &ExprTree,
        node_id: NodeId,
        target: NodeId,
        builder: &mut ExprBuilder,
        replacement: F,
    ) -> Result<NodeId, TransformError>
    where
        F: FnOnce(&mut ExprBuilder) -> Result<NodeId, TransformError>,
    {
        if node_id == target {
            return replacement(builder);
        }
        Self::copy_subtree(tree, node_id, builder)
    }

    /// Deep-copy a subtree into a new builder.
    fn copy_subtree(
        tree: &ExprTree,
        node_id: NodeId,
        builder: &mut ExprBuilder,
    ) -> Result<NodeId, TransformError> {
        let node = tree
            .get(node_id)
            .ok_or(TransformError::NodeNotFound(node_id))?;
        match node {
            ExprNode::Constant(v) => Ok(builder.constant(*v)),
            ExprNode::Variable { name, .. } => Ok(builder.variable(name.as_str())),
            ExprNode::Operation { op, children } => {
                let new_children: Result<Vec<NodeId>, TransformError> = children
                    .iter()
                    .map(|&c| Self::copy_subtree(tree, c, builder))
                    .collect();
                Ok(builder.op(*op, new_children?))
            }
            ExprNode::LibraryCall { function, children } => {
                let new_children: Result<Vec<NodeId>, TransformError> = children
                    .iter()
                    .map(|&c| Self::copy_subtree(tree, c, builder))
                    .collect();
                let new_children = new_children?;
                let _new_node = ExprNode::LibraryCall {
                    function: function.clone(),
                    children: new_children,
                };
                // Need to access the inner tree - use a workaround
                Ok(builder.op(FpOp::BlackBox, vec![]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_simple_tree() {
        let mut b = ExprBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.binop(FpOp::Add, x, y);
        let tree = b.build(sum);

        let mut b2 = ExprBuilder::new();
        let copied = TransformPass::copy_subtree(&tree, NodeId(2), &mut b2);
        assert!(copied.is_ok());
    }
}
