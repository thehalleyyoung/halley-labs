//! Expression trees for floating-point computations.
//!
//! Provides an arena-based expression tree representation used throughout
//! Penumbra to model the structure of traced computations.  Each node
//! represents either a primitive IEEE 754 operation, a constant, a variable
//! reference, or a composite (library call treated as a black box).

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── NodeId ─────────────────────────────────────────────────────────────────

/// Opaque identifier for a node in an [`ExprTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

// ─── FpOp ───────────────────────────────────────────────────────────────────

/// Primitive floating-point operations tracked by the engine.
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

    // Comparison / selection
    Min,
    Max,

    // Transcendentals
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Pow,
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

    // Rounding
    Floor,
    Ceil,
    Round,
    Trunc,

    // Reductions
    Sum,
    Prod,
    Dot,

    // Type conversion
    CastF32ToF64,
    CastF64ToF32,

    // Black-box library call (LAPACK, BLAS, etc.)
    BlackBox,
}

impl FpOp {
    /// Number of operands this operation expects.
    pub fn arity(self) -> usize {
        match self {
            Self::Neg
            | Self::Abs
            | Self::Sqrt
            | Self::Exp
            | Self::Exp2
            | Self::Log
            | Self::Log2
            | Self::Log10
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Trunc
            | Self::CastF32ToF64
            | Self::CastF64ToF32 => 1,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Min
            | Self::Max
            | Self::Pow
            | Self::Atan2
            | Self::Dot => 2,

            Self::Fma => 3,

            Self::Sum | Self::Prod | Self::BlackBox => 0, // variadic
        }
    }

    /// Whether the operation is a reduction over an array.
    pub fn is_reduction(self) -> bool {
        matches!(self, Self::Sum | Self::Prod | Self::Dot)
    }

    /// Whether this is a black-box (opaque library) call.
    pub fn is_black_box(self) -> bool {
        matches!(self, Self::BlackBox)
    }

    /// Human-readable symbol for the operation.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Fma => "fma",
            Self::Min => "min",
            Self::Max => "max",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Pow => "pow",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Trunc => "trunc",
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Dot => "dot",
            Self::CastF32ToF64 => "f32→f64",
            Self::CastF64ToF32 => "f64→f32",
            Self::BlackBox => "blackbox",
        }
    }
}

impl fmt::Display for FpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

// ─── ExprNode ───────────────────────────────────────────────────────────────

/// A node in the expression tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprNode {
    /// A floating-point constant.
    Constant(f64),
    /// A named variable (input to the traced pipeline).
    Variable { name: String, index: Option<usize> },
    /// A primitive FP operation with child operands.
    Operation { op: FpOp, children: Vec<NodeId> },
    /// A black-box library call with named function and operands.
    LibraryCall {
        function: String,
        children: Vec<NodeId>,
    },
}

// ─── ExprTree ───────────────────────────────────────────────────────────────

/// Arena-backed expression tree.
///
/// Nodes are stored in a flat `Vec` and referenced by [`NodeId`].
/// This avoids pointer chasing and simplifies serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprTree {
    nodes: Vec<ExprNode>,
    /// The root node of the tree (typically the final output).
    pub root: Option<NodeId>,
}

impl ExprTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }

    /// Insert a node and return its [`NodeId`].
    pub fn add_node(&mut self, node: ExprNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Retrieve a node by id.
    pub fn get(&self, id: NodeId) -> Option<&ExprNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Retrieve a node mutably.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut ExprNode> {
        self.nodes.get_mut(id.0 as usize)
    }

    /// Total number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all (id, node) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &ExprNode)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId(i as u32), n))
    }

    /// Return the children of a node.
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        match self.get(id) {
            Some(ExprNode::Operation { children, .. }) => children.clone(),
            Some(ExprNode::LibraryCall { children, .. }) => children.clone(),
            _ => Vec::new(),
        }
    }
}

impl Default for ExprTree {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ExprBuilder ────────────────────────────────────────────────────────────

/// Convenience builder for constructing expression trees.
pub struct ExprBuilder {
    tree: ExprTree,
}

impl ExprBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            tree: ExprTree::new(),
        }
    }

    /// Add a constant.
    pub fn constant(&mut self, value: f64) -> NodeId {
        self.tree.add_node(ExprNode::Constant(value))
    }

    /// Add a variable.
    pub fn variable(&mut self, name: impl Into<String>) -> NodeId {
        self.tree.add_node(ExprNode::Variable {
            name: name.into(),
            index: None,
        })
    }

    /// Add an operation.
    pub fn op(&mut self, op: FpOp, children: Vec<NodeId>) -> NodeId {
        self.tree.add_node(ExprNode::Operation { op, children })
    }

    /// Add a binary operation (convenience).
    pub fn binop(&mut self, op: FpOp, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.op(op, vec![lhs, rhs])
    }

    /// Add a unary operation (convenience).
    pub fn unop(&mut self, op: FpOp, operand: NodeId) -> NodeId {
        self.op(op, vec![operand])
    }

    /// Set the root and consume the builder, returning the tree.
    pub fn build(mut self, root: NodeId) -> ExprTree {
        self.tree.root = Some(root);
        self.tree
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
    fn build_simple_expression() {
        let mut b = ExprBuilder::new();
        let x = b.variable("x");
        let y = b.variable("y");
        let sum = b.binop(FpOp::Add, x, y);
        let tree = b.build(sum);
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.root, Some(NodeId(2)));
    }
}
