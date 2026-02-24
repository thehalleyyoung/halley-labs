"""
Common Refinement IR — SSA-based intermediate representation with gated φ-nodes.

Supports both Python and TypeScript frontends with a unified node set
that preserves refinement-relevant semantics.
"""

from src.ir.unified import (
    IRModule,
    IRFunction,
    IRBasicBlock,
    CFG,
    SSAValue,
    DominatorTree,
)
from src.ir.nodes import (
    IRNode,
    AssignNode,
    PhiNode,
    GuardNode,
    CallNode,
    ReturnNode,
    BranchNode,
    BinOpNode,
    UnaryOpNode,
    LoadAttrNode,
    StoreAttrNode,
    IndexNode,
    LiteralNode,
    TruthinessNode,
    TypeNarrowNode,
)

__all__ = [
    "IRModule",
    "IRFunction",
    "IRBasicBlock",
    "CFG",
    "SSAValue",
    "DominatorTree",
    "IRNode",
    "AssignNode",
    "PhiNode",
    "GuardNode",
    "CallNode",
    "ReturnNode",
    "BranchNode",
    "BinOpNode",
    "UnaryOpNode",
    "LoadAttrNode",
    "StoreAttrNode",
    "IndexNode",
    "LiteralNode",
    "TruthinessNode",
    "TypeNarrowNode",
]
