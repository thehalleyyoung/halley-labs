"""
Unified IR Module for Refinement Type Inference.

Provides a common SSA-based intermediate representation with gated φ-nodes
for both Python and TypeScript. Supports exception/generator control flow,
explicit truthiness coercion, type narrowing at guard points, and
language-tag annotations for differing semantics.
"""

from __future__ import annotations

import json
import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "LanguageTag",
    "EdgeKind",
    "SourceLocation",
    "SSAValue",
    "SSAVariable",
    "IRType",
    "RefinementType",
    "IRNode",
    "PhiNode",
    "GatedPhiNode",
    "TruthinessCoercionNode",
    "TypeNarrowNode",
    "AssignNode",
    "CallNode",
    "ReturnNode",
    "BranchNode",
    "ConditionalBranchNode",
    "YieldNode",
    "RaiseNode",
    "LoadAttrNode",
    "StoreAttrNode",
    "LoadSubscriptNode",
    "StoreSubscriptNode",
    "BinOpNode",
    "UnaryOpNode",
    "CompareNode",
    "ConstantNode",
    "ImportNode",
    "LiteralNode",
    "IRBasicBlock",
    "CFGEdge",
    "CFG",
    "DominatorTree",
    "SSABuilder",
    "PhiInserter",
    "VariableRenamer",
    "CFGBuilder",
    "IRFunction",
    "IRClass",
    "IRModule",
    "IRVisitor",
    "IRRewriter",
    "IRPrinter",
    "IRValidator",
    "IRSerializer",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LanguageTag(Enum):
    """Source language tag for differing semantic interpretation."""
    Python = "python"
    TypeScript = "typescript"


class EdgeKind(Enum):
    """Kind of control-flow edge in the CFG."""
    Normal = auto()
    Exception = auto()
    BackEdge = auto()
    TrueGuard = auto()
    FalseGuard = auto()


class BinOp(Enum):
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    FloorDiv = "//"
    Mod = "%"
    Pow = "**"
    BitAnd = "&"
    BitOr = "|"
    BitXor = "^"
    LShift = "<<"
    RShift = ">>"
    And = "and"
    Or = "or"


class UnaryOp(Enum):
    Neg = "-"
    Pos = "+"
    Not = "not"
    Invert = "~"


class CompareOp(Enum):
    Eq = "=="
    NotEq = "!="
    Lt = "<"
    LtE = "<="
    Gt = ">"
    GtE = ">="
    Is = "is"
    IsNot = "is not"
    In = "in"
    NotIn = "not in"
    Instanceof = "instanceof"


# ---------------------------------------------------------------------------
# Source Location
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Tracks a position in source code."""
    file: str = "<unknown>"
    line: int = 0
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        s = f"{self.file}:{self.line}:{self.column}"
        if self.end_line is not None:
            s += f"-{self.end_line}:{self.end_column}"
        return s


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class IRType:
    """Base IR type representation."""
    name: str
    args: List[IRType] = field(default_factory=list)
    nullable: bool = False
    language_tag: Optional[LanguageTag] = None

    def __str__(self) -> str:
        s = self.name
        if self.args:
            s += "[" + ", ".join(str(a) for a in self.args) + "]"
        if self.nullable:
            s += "?"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IRType):
            return NotImplemented
        return (
            self.name == other.name
            and self.args == other.args
            and self.nullable == other.nullable
        )

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args), self.nullable))


@dataclass
class RefinementType(IRType):
    """A refinement type {x: T | φ(x)} carrying a predicate."""
    base_type: Optional[IRType] = None
    predicate: Optional[str] = None

    def __str__(self) -> str:
        base = str(self.base_type) if self.base_type else self.name
        if self.predicate:
            return f"{{{base} | {self.predicate}}}"
        return base


# Sentinel types
TOP_TYPE = IRType("⊤")
BOTTOM_TYPE = IRType("⊥")
UNKNOWN_TYPE = IRType("unknown")
NONE_TYPE = IRType("None")
INT_TYPE = IRType("int")
FLOAT_TYPE = IRType("float")
STR_TYPE = IRType("str")
BOOL_TYPE = IRType("bool")
ANY_TYPE = IRType("Any")


# ---------------------------------------------------------------------------
# SSA Value / Variable
# ---------------------------------------------------------------------------

@dataclass
class SSAValue:
    """Represents an SSA variable with version number and type annotation."""
    name: str
    version: int = 0
    type_annotation: Optional[IRType] = None
    source_location: Optional[SourceLocation] = None
    defining_node: Optional[IRNode] = None

    @property
    def versioned_name(self) -> str:
        return f"{self.name}_{self.version}"

    def __str__(self) -> str:
        s = self.versioned_name
        if self.type_annotation:
            s += f": {self.type_annotation}"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSAValue):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __repr__(self) -> str:
        return f"SSAValue({self.versioned_name})"


@dataclass
class SSAVariable:
    """Named variable in SSA form — wraps an SSAValue with metadata."""
    ssa_value: SSAValue
    is_parameter: bool = False
    is_closure_capture: bool = False
    is_global: bool = False

    @property
    def name(self) -> str:
        return self.ssa_value.name

    @property
    def version(self) -> int:
        return self.ssa_value.version

    @property
    def versioned_name(self) -> str:
        return self.ssa_value.versioned_name

    def __str__(self) -> str:
        return str(self.ssa_value)

    def __hash__(self) -> int:
        return hash(self.ssa_value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSAVariable):
            return NotImplemented
        return self.ssa_value == other.ssa_value


# ---------------------------------------------------------------------------
# IR Nodes
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    """Base class for all IR nodes."""
    location: Optional[SourceLocation] = None
    language_tag: Optional[LanguageTag] = None
    _id: int = field(default_factory=lambda: IRNode._next_id())
    metadata: Dict[str, Any] = field(default_factory=dict)

    _counter: int = 0

    @staticmethod
    def _next_id() -> int:
        IRNode._counter += 1
        return IRNode._counter

    @staticmethod
    def reset_id_counter() -> None:
        IRNode._counter = 0

    def defined_values(self) -> List[SSAValue]:
        """Return SSA values defined by this node."""
        return []

    def used_values(self) -> List[SSAValue]:
        """Return SSA values used by this node."""
        return []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_node(self)


# --- Phi Nodes ---

@dataclass
class PhiNode(IRNode):
    """Standard φ-node: merges SSA values from predecessor blocks."""
    target: Optional[SSAValue] = None
    incoming: Dict[str, SSAValue] = field(default_factory=dict)  # block_label -> value

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return list(self.incoming.values())

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_phi(self)

    def __str__(self) -> str:
        incoming_str = ", ".join(
            f"[{lbl}: {v}]" for lbl, v in sorted(self.incoming.items())
        )
        return f"{self.target} = φ({incoming_str})"


@dataclass
class GatedPhiNode(IRNode):
    """
    Gated φ-node for exception/generator control flow.
    gate_kind distinguishes exception-handler merges, generator yields, etc.
    """
    target: Optional[SSAValue] = None
    gate_kind: str = "exception"  # "exception", "generator", "finally"
    incoming: Dict[str, SSAValue] = field(default_factory=dict)
    gate_condition: Optional[SSAValue] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        vals = list(self.incoming.values())
        if self.gate_condition:
            vals.append(self.gate_condition)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_gated_phi(self)

    def __str__(self) -> str:
        incoming_str = ", ".join(
            f"[{lbl}: {v}]" for lbl, v in sorted(self.incoming.items())
        )
        gate = f", gate={self.gate_condition}" if self.gate_condition else ""
        return f"{self.target} = γ_{self.gate_kind}({incoming_str}{gate})"


# --- Truthiness / Narrowing ---

@dataclass
class TruthinessCoercionNode(IRNode):
    """
    Explicit truthiness coercion for Python semantics.
    Converts a value to its boolean truthiness (e.g., empty list -> False).
    """
    target: Optional[SSAValue] = None
    operand: Optional[SSAValue] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.operand] if self.operand else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_truthiness_coercion(self)

    def __str__(self) -> str:
        return f"{self.target} = truthiness({self.operand})"


@dataclass
class TypeNarrowNode(IRNode):
    """
    Explicit type narrowing at guard points.
    Records the narrowed type and the guard condition that established it.
    """
    target: Optional[SSAValue] = None
    source: Optional[SSAValue] = None
    narrowed_type: Optional[IRType] = None
    guard_condition: Optional[str] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.source] if self.source else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_type_narrow(self)

    def __str__(self) -> str:
        g = f" [{self.guard_condition}]" if self.guard_condition else ""
        return f"{self.target} = narrow({self.source} -> {self.narrowed_type}){g}"


# --- Assignment / Computation ---

@dataclass
class AssignNode(IRNode):
    target: Optional[SSAValue] = None
    value: Optional[SSAValue] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.value] if self.value else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_assign(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.value}"


@dataclass
class ConstantNode(IRNode):
    target: Optional[SSAValue] = None
    value: Any = None
    type_annotation: Optional[IRType] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_constant(self)

    def __str__(self) -> str:
        return f"{self.target} = const({self.value!r})"


@dataclass
class LiteralNode(IRNode):
    target: Optional[SSAValue] = None
    literal_value: Any = None
    literal_type: Optional[IRType] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_literal(self)

    def __str__(self) -> str:
        return f"{self.target} = literal({self.literal_value!r})"


@dataclass
class BinOpNode(IRNode):
    target: Optional[SSAValue] = None
    left: Optional[SSAValue] = None
    right: Optional[SSAValue] = None
    op: BinOp = BinOp.Add

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.left:
            vals.append(self.left)
        if self.right:
            vals.append(self.right)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_binop(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.left} {self.op.value} {self.right}"


@dataclass
class UnaryOpNode(IRNode):
    target: Optional[SSAValue] = None
    operand: Optional[SSAValue] = None
    op: UnaryOp = UnaryOp.Neg

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.operand] if self.operand else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_unaryop(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.op.value} {self.operand}"


@dataclass
class CompareNode(IRNode):
    target: Optional[SSAValue] = None
    left: Optional[SSAValue] = None
    right: Optional[SSAValue] = None
    op: CompareOp = CompareOp.Eq

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.left:
            vals.append(self.left)
        if self.right:
            vals.append(self.right)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_compare(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.left} {self.op.value} {self.right}"


# --- Call ---

@dataclass
class CallNode(IRNode):
    target: Optional[SSAValue] = None
    callee: Optional[SSAValue] = None
    args: List[SSAValue] = field(default_factory=list)
    kwargs: Dict[str, SSAValue] = field(default_factory=dict)
    is_new: bool = False  # TypeScript `new` call

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.callee:
            vals.append(self.callee)
        vals.extend(self.args)
        vals.extend(self.kwargs.values())
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_call(self)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        kw_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kw_str]))
        prefix = "new " if self.is_new else ""
        return f"{self.target} = {prefix}{self.callee}({all_args})"


# --- Attribute / Subscript ---

@dataclass
class LoadAttrNode(IRNode):
    target: Optional[SSAValue] = None
    obj: Optional[SSAValue] = None
    attr: str = ""

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.obj] if self.obj else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_load_attr(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.obj}.{self.attr}"


@dataclass
class StoreAttrNode(IRNode):
    obj: Optional[SSAValue] = None
    attr: str = ""
    value: Optional[SSAValue] = None

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.obj:
            vals.append(self.obj)
        if self.value:
            vals.append(self.value)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_store_attr(self)

    def __str__(self) -> str:
        return f"{self.obj}.{self.attr} = {self.value}"


@dataclass
class LoadSubscriptNode(IRNode):
    target: Optional[SSAValue] = None
    obj: Optional[SSAValue] = None
    index: Optional[SSAValue] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.obj:
            vals.append(self.obj)
        if self.index:
            vals.append(self.index)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_load_subscript(self)

    def __str__(self) -> str:
        return f"{self.target} = {self.obj}[{self.index}]"


@dataclass
class StoreSubscriptNode(IRNode):
    obj: Optional[SSAValue] = None
    index: Optional[SSAValue] = None
    value: Optional[SSAValue] = None

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.obj:
            vals.append(self.obj)
        if self.index:
            vals.append(self.index)
        if self.value:
            vals.append(self.value)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_store_subscript(self)

    def __str__(self) -> str:
        return f"{self.obj}[{self.index}] = {self.value}"


# --- Control flow terminators ---

@dataclass
class ReturnNode(IRNode):
    value: Optional[SSAValue] = None

    def used_values(self) -> List[SSAValue]:
        return [self.value] if self.value else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_return(self)

    def __str__(self) -> str:
        return f"return {self.value}" if self.value else "return"


@dataclass
class BranchNode(IRNode):
    target_label: str = ""

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_branch(self)

    def __str__(self) -> str:
        return f"br {self.target_label}"


@dataclass
class ConditionalBranchNode(IRNode):
    condition: Optional[SSAValue] = None
    true_label: str = ""
    false_label: str = ""

    def used_values(self) -> List[SSAValue]:
        return [self.condition] if self.condition else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_cond_branch(self)

    def __str__(self) -> str:
        return f"br {self.condition}, {self.true_label}, {self.false_label}"


@dataclass
class YieldNode(IRNode):
    """Yield / yield-from for generators, await for async."""
    target: Optional[SSAValue] = None
    value: Optional[SSAValue] = None
    is_yield_from: bool = False
    is_await: bool = False

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def used_values(self) -> List[SSAValue]:
        return [self.value] if self.value else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_yield(self)

    def __str__(self) -> str:
        kw = "await" if self.is_await else ("yield from" if self.is_yield_from else "yield")
        t = f"{self.target} = " if self.target else ""
        return f"{t}{kw} {self.value}"


@dataclass
class RaiseNode(IRNode):
    exception: Optional[SSAValue] = None
    cause: Optional[SSAValue] = None

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        if self.exception:
            vals.append(self.exception)
        if self.cause:
            vals.append(self.cause)
        return vals

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_raise(self)

    def __str__(self) -> str:
        s = "raise"
        if self.exception:
            s += f" {self.exception}"
        if self.cause:
            s += f" from {self.cause}"
        return s


@dataclass
class ImportNode(IRNode):
    target: Optional[SSAValue] = None
    module_name: str = ""
    imported_name: Optional[str] = None
    alias: Optional[str] = None

    def defined_values(self) -> List[SSAValue]:
        return [self.target] if self.target else []

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_import(self)

    def __str__(self) -> str:
        if self.imported_name:
            return f"{self.target} = from {self.module_name} import {self.imported_name}"
        return f"{self.target} = import {self.module_name}"


# ---------------------------------------------------------------------------
# CFG Edge
# ---------------------------------------------------------------------------

@dataclass
class CFGEdge:
    """An edge in the control flow graph."""
    source: str
    target: str
    kind: EdgeKind = EdgeKind.Normal
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.source} -[{self.kind.name}]-> {self.target}"


# ---------------------------------------------------------------------------
# Basic Block
# ---------------------------------------------------------------------------

@dataclass
class IRBasicBlock:
    """A basic block with label, IR nodes, terminator, and graph info."""
    label: str
    nodes: List[IRNode] = field(default_factory=list)
    terminator: Optional[IRNode] = None
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    # Dominator info (filled by DominatorTree)
    idom: Optional[str] = None
    dominance_frontier: Set[str] = field(default_factory=set)
    dom_children: List[str] = field(default_factory=list)
    dom_depth: int = 0

    @property
    def phi_nodes(self) -> List[PhiNode]:
        return [n for n in self.nodes if isinstance(n, (PhiNode, GatedPhiNode))]

    @property
    def non_phi_nodes(self) -> List[IRNode]:
        return [n for n in self.nodes if not isinstance(n, (PhiNode, GatedPhiNode))]

    def add_node(self, node: IRNode) -> None:
        self.nodes.append(node)

    def insert_phi(self, phi: PhiNode) -> None:
        """Insert a phi node at the beginning of the block (after existing phis)."""
        idx = 0
        for i, n in enumerate(self.nodes):
            if isinstance(n, (PhiNode, GatedPhiNode)):
                idx = i + 1
            else:
                break
        self.nodes.insert(idx, phi)

    def all_instructions(self) -> Iterator[IRNode]:
        yield from self.nodes
        if self.terminator:
            yield self.terminator

    def defined_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        for inst in self.all_instructions():
            vals.extend(inst.defined_values())
        return vals

    def used_values(self) -> List[SSAValue]:
        vals: List[SSAValue] = []
        for inst in self.all_instructions():
            vals.extend(inst.used_values())
        return vals

    def __str__(self) -> str:
        lines = [f"{self.label}:"]
        for n in self.nodes:
            lines.append(f"  {n}")
        if self.terminator:
            lines.append(f"  {self.terminator}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Control Flow Graph
# ---------------------------------------------------------------------------

@dataclass
class CFG:
    """
    Control flow graph with entry/exit blocks, block map, edge types,
    and loop detection.
    """
    entry_label: str = "entry"
    exit_label: str = "exit"
    blocks: Dict[str, IRBasicBlock] = field(default_factory=dict)
    edges: List[CFGEdge] = field(default_factory=list)

    # Computed
    _back_edges: Set[Tuple[str, str]] = field(default_factory=set)
    _loop_headers: Set[str] = field(default_factory=set)
    _loops: Dict[str, Set[str]] = field(default_factory=dict)  # header -> body blocks

    @property
    def entry_block(self) -> Optional[IRBasicBlock]:
        return self.blocks.get(self.entry_label)

    @property
    def exit_block(self) -> Optional[IRBasicBlock]:
        return self.blocks.get(self.exit_label)

    def add_block(self, block: IRBasicBlock) -> None:
        self.blocks[block.label] = block

    def add_edge(self, source: str, target: str, kind: EdgeKind = EdgeKind.Normal) -> None:
        edge = CFGEdge(source=source, target=target, kind=kind)
        self.edges.append(edge)
        if source in self.blocks:
            if target not in self.blocks[source].successors:
                self.blocks[source].successors.append(target)
        if target in self.blocks:
            if source not in self.blocks[target].predecessors:
                self.blocks[target].predecessors.append(source)

    def remove_edge(self, source: str, target: str) -> None:
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        if source in self.blocks and target in self.blocks[source].successors:
            self.blocks[source].successors.remove(target)
        if target in self.blocks and source in self.blocks[target].predecessors:
            self.blocks[target].predecessors.remove(source)

    def get_edges(self, source: str, target: str) -> List[CFGEdge]:
        return [e for e in self.edges if e.source == source and e.target == target]

    def reverse_postorder(self) -> List[str]:
        """Return block labels in reverse postorder."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(label: str) -> None:
            if label in visited or label not in self.blocks:
                return
            visited.add(label)
            for succ in self.blocks[label].successors:
                dfs(succ)
            order.append(label)

        dfs(self.entry_label)
        order.reverse()
        return order

    def postorder(self) -> List[str]:
        """Return block labels in postorder."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(label: str) -> None:
            if label in visited or label not in self.blocks:
                return
            visited.add(label)
            for succ in self.blocks[label].successors:
                dfs(succ)
            order.append(label)

        dfs(self.entry_label)
        return order

    def detect_loops(self) -> None:
        """Detect natural loops via back-edge identification."""
        self._back_edges.clear()
        self._loop_headers.clear()
        self._loops.clear()

        # Compute DFS tree to find back edges
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(label: str) -> None:
            if label not in self.blocks:
                return
            visited.add(label)
            in_stack.add(label)
            for succ in self.blocks[label].successors:
                if succ in in_stack:
                    self._back_edges.add((label, succ))
                    self._loop_headers.add(succ)
                    # Mark the edge
                    for e in self.get_edges(label, succ):
                        e.kind = EdgeKind.BackEdge
                elif succ not in visited:
                    dfs(succ)
            in_stack.discard(label)

        dfs(self.entry_label)

        # For each back edge, compute the natural loop body
        for tail, header in self._back_edges:
            body: Set[str] = {header}
            if tail != header:
                body.add(tail)
                worklist: Deque[str] = deque([tail])
                while worklist:
                    node = worklist.popleft()
                    for pred in self.blocks[node].predecessors:
                        if pred not in body and pred in self.blocks:
                            body.add(pred)
                            worklist.append(pred)
            if header in self._loops:
                self._loops[header] |= body
            else:
                self._loops[header] = body

    @property
    def back_edges(self) -> Set[Tuple[str, str]]:
        return self._back_edges

    @property
    def loop_headers(self) -> Set[str]:
        return self._loop_headers

    @property
    def loops(self) -> Dict[str, Set[str]]:
        return self._loops

    def reachable_blocks(self) -> Set[str]:
        """Return set of labels reachable from entry."""
        visited: Set[str] = set()
        worklist: Deque[str] = deque([self.entry_label])
        while worklist:
            lbl = worklist.popleft()
            if lbl in visited or lbl not in self.blocks:
                continue
            visited.add(lbl)
            for succ in self.blocks[lbl].successors:
                worklist.append(succ)
        return visited

    def remove_unreachable(self) -> Set[str]:
        """Remove unreachable blocks, return set of removed labels."""
        reachable = self.reachable_blocks()
        unreachable = set(self.blocks.keys()) - reachable
        for lbl in unreachable:
            del self.blocks[lbl]
        self.edges = [e for e in self.edges if e.source in reachable and e.target in reachable]
        for blk in self.blocks.values():
            blk.predecessors = [p for p in blk.predecessors if p in reachable]
            blk.successors = [s for s in blk.successors if s in reachable]
        return unreachable

    def __str__(self) -> str:
        rpo = self.reverse_postorder()
        remaining = [l for l in self.blocks if l not in rpo]
        lines: List[str] = []
        for lbl in rpo + remaining:
            if lbl in self.blocks:
                lines.append(str(self.blocks[lbl]))
                lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dominator Tree — Lengauer-Tarjan Algorithm
# ---------------------------------------------------------------------------

class DominatorTree:
    """
    Dominator tree computation using the Lengauer-Tarjan algorithm.
    Also computes dominance frontiers for SSA φ-node insertion.
    """

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self._idom: Dict[str, Optional[str]] = {}
        self._dom_tree: Dict[str, List[str]] = defaultdict(list)
        self._dom_frontier: Dict[str, Set[str]] = defaultdict(set)
        self._depth: Dict[str, int] = {}

    def compute(self) -> None:
        """Compute dominator tree using Lengauer-Tarjan and then dominance frontiers."""
        if not self.cfg.blocks:
            return

        entry = self.cfg.entry_label
        if entry not in self.cfg.blocks:
            return

        # --- Lengauer-Tarjan ---
        # Step 1: DFS numbering
        dfs_num: Dict[str, int] = {}
        dfs_vertex: List[str] = []
        dfs_parent: Dict[str, Optional[str]] = {}
        semi: Dict[str, int] = {}
        idom: Dict[str, Optional[str]] = {}
        ancestor: Dict[str, Optional[str]] = {}
        label: Dict[str, str] = {}
        bucket: Dict[str, Set[str]] = defaultdict(set)

        n = 0
        stack: List[Tuple[str, Optional[str], bool]] = [(entry, None, False)]
        visited_dfs: Set[str] = set()

        while stack:
            node, parent, processed = stack.pop()
            if processed:
                continue
            if node in visited_dfs:
                continue
            visited_dfs.add(node)
            dfs_num[node] = n
            dfs_vertex.append(node)
            semi[node] = n
            label[node] = node
            ancestor[node] = None
            dfs_parent[node] = parent
            n += 1

            block = self.cfg.blocks.get(node)
            if block:
                for succ in reversed(block.successors):
                    if succ not in visited_dfs and succ in self.cfg.blocks:
                        stack.append((succ, node, False))

        if n == 0:
            return

        # Helper: path compression
        def _compress(v: str) -> None:
            a = ancestor[v]
            if a is not None and ancestor[a] is not None:
                _compress(a)
                if semi[label[a]] < semi[label[v]]:
                    label[v] = label[a]
                ancestor[v] = ancestor[a]

        def _eval(v: str) -> str:
            if ancestor[v] is None:
                return v
            _compress(v)
            return label[v]

        def _link(v: str, w: str) -> None:
            ancestor[w] = v

        # Step 2-3: Compute semidominators and implicitly fill buckets
        for i in range(n - 1, 0, -1):
            w = dfs_vertex[i]
            block_w = self.cfg.blocks.get(w)
            if block_w:
                for v in block_w.predecessors:
                    if v in dfs_num:
                        u = _eval(v)
                        if semi[u] < semi[w]:
                            semi[w] = semi[u]

            bucket[dfs_vertex[semi[w]]].add(w)
            p = dfs_parent[w]
            if p is not None:
                _link(p, w)

                for v in list(bucket[p]):
                    bucket[p].discard(v)
                    u = _eval(v)
                    idom[v] = u if semi[u] < semi[v] else p

        # Step 4: Finalize idoms
        for i in range(1, n):
            w = dfs_vertex[i]
            if w in idom and idom[w] != dfs_vertex[semi[w]]:
                idom[w] = idom.get(idom[w])

        idom[entry] = None
        self._idom = idom

        # Build dominator tree children
        self._dom_tree.clear()
        for node, parent in self._idom.items():
            if parent is not None:
                self._dom_tree[parent].append(node)

        # Compute depths
        self._depth.clear()
        self._depth[entry] = 0
        worklist: Deque[str] = deque([entry])
        while worklist:
            node = worklist.popleft()
            for child in self._dom_tree[node]:
                self._depth[child] = self._depth[node] + 1
                worklist.append(child)

        # Annotate blocks
        for lbl, blk in self.cfg.blocks.items():
            blk.idom = self._idom.get(lbl)
            blk.dom_children = list(self._dom_tree.get(lbl, []))
            blk.dom_depth = self._depth.get(lbl, 0)

        # Compute dominance frontiers (Cooper, Harvey, Kennedy)
        self._compute_dominance_frontiers()

    def _compute_dominance_frontiers(self) -> None:
        """Compute dominance frontiers using the DJ-graph approach."""
        self._dom_frontier.clear()

        for lbl in self.cfg.blocks:
            self._dom_frontier[lbl] = set()

        for lbl, blk in self.cfg.blocks.items():
            if len(blk.predecessors) >= 2:
                for pred in blk.predecessors:
                    runner = pred
                    while runner is not None and runner != self._idom.get(lbl):
                        self._dom_frontier[runner].add(lbl)
                        runner = self._idom.get(runner)

        for lbl, blk in self.cfg.blocks.items():
            blk.dominance_frontier = self._dom_frontier[lbl]

    @property
    def idom(self) -> Dict[str, Optional[str]]:
        return self._idom

    @property
    def dom_tree(self) -> Dict[str, List[str]]:
        return dict(self._dom_tree)

    @property
    def dom_frontier(self) -> Dict[str, Set[str]]:
        return dict(self._dom_frontier)

    def dominates(self, a: str, b: str) -> bool:
        """Return True if block a dominates block b."""
        if a == b:
            return True
        runner: Optional[str] = b
        while runner is not None:
            runner = self._idom.get(runner)
            if runner == a:
                return True
        return False

    def lca(self, a: str, b: str) -> Optional[str]:
        """Lowest common ancestor in the dominator tree."""
        ancestors_a: Set[str] = set()
        runner: Optional[str] = a
        while runner is not None:
            ancestors_a.add(runner)
            runner = self._idom.get(runner)
        runner = b
        while runner is not None:
            if runner in ancestors_a:
                return runner
            runner = self._idom.get(runner)
        return None


# ---------------------------------------------------------------------------
# Phi Inserter (Iterated Dominance Frontier)
# ---------------------------------------------------------------------------

class PhiInserter:
    """
    Insert φ-functions at iterated dominance frontiers.

    For each variable, find all blocks where it is defined, compute the
    iterated dominance frontier of those blocks, and insert φ-nodes.
    """

    def __init__(self, cfg: CFG, dom_tree: DominatorTree) -> None:
        self.cfg = cfg
        self.dom_tree = dom_tree

    def insert(self, var_defs: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Insert phi nodes for variables.

        Args:
            var_defs: mapping from variable name to set of block labels
                      where the variable is defined.

        Returns:
            mapping from variable name to set of block labels where
            φ-nodes were inserted.
        """
        phi_locations: Dict[str, Set[str]] = defaultdict(set)

        for var_name, def_blocks in var_defs.items():
            # Iterated dominance frontier
            idf = self._iterated_dominance_frontier(def_blocks)
            for lbl in idf:
                if lbl in self.cfg.blocks:
                    phi = PhiNode(
                        target=SSAValue(name=var_name, version=0),
                        incoming={},
                    )
                    self.cfg.blocks[lbl].insert_phi(phi)
                    phi_locations[var_name].add(lbl)

        return dict(phi_locations)

    def _iterated_dominance_frontier(self, blocks: Set[str]) -> Set[str]:
        """Compute the iterated dominance frontier of a set of blocks."""
        idf: Set[str] = set()
        worklist: Deque[str] = deque(blocks)
        ever_on_worklist: Set[str] = set(blocks)

        while worklist:
            x = worklist.popleft()
            df_x = self.dom_tree.dom_frontier.get(x, set())
            for y in df_x:
                if y not in idf:
                    idf.add(y)
                    if y not in ever_on_worklist:
                        ever_on_worklist.add(y)
                        worklist.append(y)

        return idf


# ---------------------------------------------------------------------------
# Variable Renamer (stack-based)
# ---------------------------------------------------------------------------

class VariableRenamer:
    """
    Stack-based variable renaming pass for SSA construction.

    Walks the dominator tree in preorder, pushing new versions onto
    per-variable stacks and popping when leaving a subtree.
    """

    def __init__(self, cfg: CFG, dom_tree: DominatorTree) -> None:
        self.cfg = cfg
        self.dom_tree = dom_tree
        self._counter: Dict[str, int] = defaultdict(int)
        self._stacks: Dict[str, List[int]] = defaultdict(list)

    def rename(self, initial_defs: Optional[Dict[str, int]] = None) -> None:
        """
        Rename all variables in the CFG to unique SSA versions.

        Args:
            initial_defs: optional mapping of variable names to initial
                          version numbers (e.g., for function parameters).
        """
        if initial_defs:
            for var, ver in initial_defs.items():
                self._counter[var] = ver
                self._stacks[var].append(ver)

        entry = self.cfg.entry_label
        if entry in self.cfg.blocks:
            self._rename_block(entry)

    def _new_version(self, name: str) -> int:
        ver = self._counter[name]
        self._counter[name] = ver + 1
        self._stacks[name].append(ver)
        return ver

    def _current_version(self, name: str) -> int:
        if self._stacks[name]:
            return self._stacks[name][-1]
        return self._new_version(name)

    def _rename_block(self, label: str) -> None:
        block = self.cfg.blocks[label]
        push_counts: Dict[str, int] = defaultdict(int)

        # Rename definitions in phi nodes
        for node in block.nodes:
            if isinstance(node, (PhiNode, GatedPhiNode)):
                if node.target:
                    ver = self._new_version(node.target.name)
                    node.target.version = ver
                    push_counts[node.target.name] += 1

        # Rename uses and definitions in regular instructions
        for node in block.nodes:
            if isinstance(node, (PhiNode, GatedPhiNode)):
                continue
            for val in node.used_values():
                val.version = self._current_version(val.name)
            for val in node.defined_values():
                ver = self._new_version(val.name)
                val.version = ver
                push_counts[val.name] += 1

        # Rename uses in terminator
        if block.terminator:
            for val in block.terminator.used_values():
                val.version = self._current_version(val.name)
            for val in block.terminator.defined_values():
                ver = self._new_version(val.name)
                val.version = ver
                push_counts[val.name] += 1

        # Fill phi operands in successor blocks
        for succ_label in block.successors:
            succ = self.cfg.blocks.get(succ_label)
            if not succ:
                continue
            for node in succ.nodes:
                if isinstance(node, (PhiNode, GatedPhiNode)):
                    if node.target:
                        var_name = node.target.name
                        if self._stacks[var_name]:
                            ver = self._stacks[var_name][-1]
                            node.incoming[label] = SSAValue(
                                name=var_name, version=ver
                            )

        # Recurse into dominator tree children
        for child in self.dom_tree.dom_tree.get(label, []):
            self._rename_block(child)

        # Pop the versions pushed in this block
        for var, count in push_counts.items():
            for _ in range(count):
                self._stacks[var].pop()


# ---------------------------------------------------------------------------
# SSA Builder — complete SSA construction
# ---------------------------------------------------------------------------

class SSABuilder:
    """
    Complete SSA construction combining phi insertion and variable renaming.

    Algorithm:
    1. Collect variable definitions per block.
    2. Compute dominator tree and dominance frontiers.
    3. Insert phi nodes at iterated dominance frontiers.
    4. Rename variables using stack-based renaming.
    """

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.dom_tree = DominatorTree(cfg)
        self._phi_locations: Dict[str, Set[str]] = {}

    def build(self, initial_defs: Optional[Dict[str, int]] = None) -> None:
        """Build SSA form for the CFG."""
        # Step 1: Compute dominators
        self.dom_tree.compute()

        # Step 2: Collect variable definitions
        var_defs = self._collect_var_defs()

        # Step 3: Insert phi nodes
        inserter = PhiInserter(self.cfg, self.dom_tree)
        self._phi_locations = inserter.insert(var_defs)

        # Step 4: Rename variables
        renamer = VariableRenamer(self.cfg, self.dom_tree)
        renamer.rename(initial_defs)

    def _collect_var_defs(self) -> Dict[str, Set[str]]:
        """Collect the set of blocks where each variable is defined."""
        var_defs: Dict[str, Set[str]] = defaultdict(set)
        for lbl, blk in self.cfg.blocks.items():
            for inst in blk.all_instructions():
                for val in inst.defined_values():
                    var_defs[val.name].add(lbl)
        return dict(var_defs)

    @property
    def phi_locations(self) -> Dict[str, Set[str]]:
        return self._phi_locations


# ---------------------------------------------------------------------------
# CFG Builder
# ---------------------------------------------------------------------------

class CFGBuilder:
    """Utility to incrementally build a CFG from basic blocks."""

    def __init__(self, entry_label: str = "entry", exit_label: str = "exit") -> None:
        self.cfg = CFG(entry_label=entry_label, exit_label=exit_label)
        self._current_block: Optional[IRBasicBlock] = None

    def new_block(self, label: str) -> IRBasicBlock:
        block = IRBasicBlock(label=label)
        self.cfg.add_block(block)
        return block

    def set_current_block(self, label: str) -> None:
        self._current_block = self.cfg.blocks.get(label)

    @property
    def current_block(self) -> Optional[IRBasicBlock]:
        return self._current_block

    def emit(self, node: IRNode) -> None:
        if self._current_block is None:
            raise RuntimeError("No current block set")
        self._current_block.add_node(node)

    def set_terminator(self, node: IRNode) -> None:
        if self._current_block is None:
            raise RuntimeError("No current block set")
        self._current_block.terminator = node

    def add_edge(self, source: str, target: str, kind: EdgeKind = EdgeKind.Normal) -> None:
        self.cfg.add_edge(source, target, kind)

    def connect(self, source: str, target: str, kind: EdgeKind = EdgeKind.Normal) -> None:
        """Alias for add_edge."""
        self.add_edge(source, target, kind)

    def finalize(self) -> CFG:
        """Finalize the CFG — detect loops and return."""
        self.cfg.detect_loops()
        return self.cfg


# ---------------------------------------------------------------------------
# IR Function
# ---------------------------------------------------------------------------

@dataclass
class IRFunction:
    """
    Function representation with params (with refinement types), return type,
    body as CFG, local SSA variables, closure captures, etc.
    """
    name: str
    params: List[SSAVariable] = field(default_factory=list)
    return_type: Optional[IRType] = None
    body: Optional[CFG] = None
    local_variables: Dict[str, SSAVariable] = field(default_factory=dict)
    closure_captures: List[SSAVariable] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    decorators: List[str] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    language_tag: Optional[LanguageTag] = None
    type_params: List[str] = field(default_factory=list)  # generic type parameters
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        prefix = ""
        if self.is_async:
            prefix += "async "
        if self.is_generator:
            prefix += "generator "
        return f"{prefix}{self.name}"

    def __str__(self) -> str:
        params_str = ", ".join(str(p) for p in self.params)
        ret = f" -> {self.return_type}" if self.return_type else ""
        header = f"def {self.qualified_name}({params_str}){ret}"
        if self.decorators:
            dec = "\n".join(f"@{d}" for d in self.decorators)
            header = f"{dec}\n{header}"
        if self.body:
            return f"{header}:\n{self.body}"
        return header


# ---------------------------------------------------------------------------
# IR Class
# ---------------------------------------------------------------------------

@dataclass
class IRClass:
    """Class with methods, fields, bases, and MRO."""
    name: str
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, IRFunction] = field(default_factory=dict)
    fields: Dict[str, IRType] = field(default_factory=dict)
    static_methods: Dict[str, IRFunction] = field(default_factory=dict)
    class_methods: Dict[str, IRFunction] = field(default_factory=dict)
    properties: Dict[str, IRFunction] = field(default_factory=dict)
    mro: List[str] = field(default_factory=list)
    is_abstract: bool = False
    decorators: List[str] = field(default_factory=list)
    type_params: List[str] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    language_tag: Optional[LanguageTag] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_mro(self, class_registry: Optional[Dict[str, IRClass]] = None) -> List[str]:
        """Compute C3 linearization MRO."""
        if not self.bases:
            self.mro = [self.name]
            return self.mro

        if class_registry is None:
            self.mro = [self.name] + self.bases
            return self.mro

        # C3 linearization
        linearizations: List[List[str]] = []
        for base_name in self.bases:
            base = class_registry.get(base_name)
            if base and base.mro:
                linearizations.append(list(base.mro))
            else:
                linearizations.append([base_name])
        linearizations.append(list(self.bases))

        result = [self.name]
        while linearizations:
            # Find a good head
            head: Optional[str] = None
            for lin in linearizations:
                if not lin:
                    continue
                candidate = lin[0]
                # Check candidate is not in the tail of any linearization
                in_tail = any(candidate in l[1:] for l in linearizations)
                if not in_tail:
                    head = candidate
                    break

            if head is None:
                raise TypeError(
                    f"Cannot compute consistent MRO for {self.name} "
                    f"with bases {self.bases}"
                )

            result.append(head)
            # Remove head from all linearizations
            linearizations = [
                [x for x in lin if x != head] for lin in linearizations
            ]
            linearizations = [lin for lin in linearizations if lin]

        self.mro = result
        return self.mro

    def all_methods(self) -> Dict[str, IRFunction]:
        """Return all methods including static and class methods."""
        result = dict(self.methods)
        result.update(self.static_methods)
        result.update(self.class_methods)
        return result

    def __str__(self) -> str:
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines = [f"class {self.name}{bases_str}:"]
        for fname, ftype in self.fields.items():
            lines.append(f"  {fname}: {ftype}")
        for mname, method in self.methods.items():
            lines.append(f"  {method}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR Module
# ---------------------------------------------------------------------------

@dataclass
class IRModule:
    """
    Top-level container: functions, classes, globals, imports,
    with source language tag.
    """
    name: str = "<module>"
    source_language: LanguageTag = LanguageTag.Python
    source_file: str = "<unknown>"
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    classes: Dict[str, IRClass] = field(default_factory=dict)
    globals: Dict[str, SSAVariable] = field(default_factory=dict)
    imports: List[ImportNode] = field(default_factory=list)
    top_level_stmts: List[IRNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_function(self, func: IRFunction) -> None:
        self.functions[func.name] = func

    def add_class(self, cls: IRClass) -> None:
        self.classes[cls.name] = cls

    def add_global(self, var: SSAVariable) -> None:
        self.globals[var.name] = var

    def add_import(self, imp: ImportNode) -> None:
        self.imports.append(imp)

    def __str__(self) -> str:
        lines = [f"module {self.name} [{self.source_language.value}]"]
        lines.append("")
        for imp in self.imports:
            lines.append(str(imp))
        if self.imports:
            lines.append("")
        for gname, gvar in self.globals.items():
            lines.append(f"global {gvar}")
        if self.globals:
            lines.append("")
        for cls in self.classes.values():
            lines.append(str(cls))
            lines.append("")
        for func in self.functions.values():
            lines.append(str(func))
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR Visitor
# ---------------------------------------------------------------------------

T = TypeVar("T")


class IRVisitor:
    """Visitor pattern base class for IR traversal."""

    def visit_module(self, module: IRModule) -> Any:
        for imp in module.imports:
            imp.accept(self)
        for cls in module.classes.values():
            self.visit_class(cls)
        for func in module.functions.values():
            self.visit_function(func)
        for stmt in module.top_level_stmts:
            stmt.accept(self)

    def visit_class(self, cls: IRClass) -> Any:
        for method in cls.methods.values():
            self.visit_function(method)
        for method in cls.static_methods.values():
            self.visit_function(method)
        for method in cls.class_methods.values():
            self.visit_function(method)

    def visit_function(self, func: IRFunction) -> Any:
        if func.body:
            self.visit_cfg(func.body)

    def visit_cfg(self, cfg: CFG) -> Any:
        for lbl in cfg.reverse_postorder():
            if lbl in cfg.blocks:
                self.visit_block(cfg.blocks[lbl])

    def visit_block(self, block: IRBasicBlock) -> Any:
        for node in block.nodes:
            node.accept(self)
        if block.terminator:
            block.terminator.accept(self)

    def visit_node(self, node: IRNode) -> Any:
        pass

    def visit_phi(self, node: PhiNode) -> Any:
        return self.visit_node(node)

    def visit_gated_phi(self, node: GatedPhiNode) -> Any:
        return self.visit_node(node)

    def visit_truthiness_coercion(self, node: TruthinessCoercionNode) -> Any:
        return self.visit_node(node)

    def visit_type_narrow(self, node: TypeNarrowNode) -> Any:
        return self.visit_node(node)

    def visit_assign(self, node: AssignNode) -> Any:
        return self.visit_node(node)

    def visit_constant(self, node: ConstantNode) -> Any:
        return self.visit_node(node)

    def visit_literal(self, node: LiteralNode) -> Any:
        return self.visit_node(node)

    def visit_binop(self, node: BinOpNode) -> Any:
        return self.visit_node(node)

    def visit_unaryop(self, node: UnaryOpNode) -> Any:
        return self.visit_node(node)

    def visit_compare(self, node: CompareNode) -> Any:
        return self.visit_node(node)

    def visit_call(self, node: CallNode) -> Any:
        return self.visit_node(node)

    def visit_load_attr(self, node: LoadAttrNode) -> Any:
        return self.visit_node(node)

    def visit_store_attr(self, node: StoreAttrNode) -> Any:
        return self.visit_node(node)

    def visit_load_subscript(self, node: LoadSubscriptNode) -> Any:
        return self.visit_node(node)

    def visit_store_subscript(self, node: StoreSubscriptNode) -> Any:
        return self.visit_node(node)

    def visit_return(self, node: ReturnNode) -> Any:
        return self.visit_node(node)

    def visit_branch(self, node: BranchNode) -> Any:
        return self.visit_node(node)

    def visit_cond_branch(self, node: ConditionalBranchNode) -> Any:
        return self.visit_node(node)

    def visit_yield(self, node: YieldNode) -> Any:
        return self.visit_node(node)

    def visit_raise(self, node: RaiseNode) -> Any:
        return self.visit_node(node)

    def visit_import(self, node: ImportNode) -> Any:
        return self.visit_node(node)


# ---------------------------------------------------------------------------
# IR Rewriter
# ---------------------------------------------------------------------------

class IRRewriter(IRVisitor):
    """
    Rewriter base class for IR transformations.
    Override visit methods to return replacement nodes; None means delete.
    Default behaviour is identity (return original node).
    """

    def rewrite_cfg(self, cfg: CFG) -> CFG:
        """Rewrite all blocks in a CFG, returning a new CFG."""
        new_cfg = CFG(
            entry_label=cfg.entry_label,
            exit_label=cfg.exit_label,
        )
        for lbl in cfg.reverse_postorder():
            if lbl in cfg.blocks:
                new_block = self.rewrite_block(cfg.blocks[lbl])
                if new_block:
                    new_cfg.add_block(new_block)

        # Copy edges
        for edge in cfg.edges:
            if edge.source in new_cfg.blocks and edge.target in new_cfg.blocks:
                new_cfg.add_edge(edge.source, edge.target, edge.kind)

        return new_cfg

    def rewrite_block(self, block: IRBasicBlock) -> Optional[IRBasicBlock]:
        new_block = IRBasicBlock(
            label=block.label,
            predecessors=list(block.predecessors),
            successors=list(block.successors),
        )
        for node in block.nodes:
            result = node.accept(self)
            if result is not None:
                new_block.add_node(result if isinstance(result, IRNode) else node)
        if block.terminator:
            result = block.terminator.accept(self)
            if result is not None:
                new_block.terminator = result if isinstance(result, IRNode) else block.terminator
        return new_block

    def rewrite_function(self, func: IRFunction) -> IRFunction:
        new_func = IRFunction(
            name=func.name,
            params=list(func.params),
            return_type=func.return_type,
            local_variables=dict(func.local_variables),
            closure_captures=list(func.closure_captures),
            is_async=func.is_async,
            is_generator=func.is_generator,
            decorators=list(func.decorators),
            location=func.location,
            language_tag=func.language_tag,
            type_params=list(func.type_params),
            metadata=dict(func.metadata),
        )
        if func.body:
            new_func.body = self.rewrite_cfg(func.body)
        return new_func

    def rewrite_module(self, module: IRModule) -> IRModule:
        new_module = IRModule(
            name=module.name,
            source_language=module.source_language,
            source_file=module.source_file,
            metadata=dict(module.metadata),
        )
        new_module.imports = list(module.imports)
        new_module.globals = dict(module.globals)
        for name, func in module.functions.items():
            new_module.functions[name] = self.rewrite_function(func)
        for name, cls in module.classes.items():
            new_cls = IRClass(
                name=cls.name,
                bases=list(cls.bases),
                fields=dict(cls.fields),
                mro=list(cls.mro),
                is_abstract=cls.is_abstract,
                decorators=list(cls.decorators),
                type_params=list(cls.type_params),
                location=cls.location,
                language_tag=cls.language_tag,
                metadata=dict(cls.metadata),
            )
            for mname, method in cls.methods.items():
                new_cls.methods[mname] = self.rewrite_function(method)
            for mname, method in cls.static_methods.items():
                new_cls.static_methods[mname] = self.rewrite_function(method)
            for mname, method in cls.class_methods.items():
                new_cls.class_methods[mname] = self.rewrite_function(method)
            new_module.classes[name] = new_cls
        return new_module


# ---------------------------------------------------------------------------
# IR Printer
# ---------------------------------------------------------------------------

class IRPrinter(IRVisitor):
    """Human-readable pretty printer for the IR."""

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent
        self._level = 0
        self._lines: List[str] = []

    def _emit(self, text: str) -> None:
        prefix = " " * (self._indent * self._level)
        self._lines.append(f"{prefix}{text}")

    def _indent_inc(self) -> None:
        self._level += 1

    def _indent_dec(self) -> None:
        self._level = max(0, self._level - 1)

    def print(self, ir: Union[IRModule, IRFunction, CFG, IRBasicBlock]) -> str:
        self._lines.clear()
        self._level = 0
        if isinstance(ir, IRModule):
            self.visit_module(ir)
        elif isinstance(ir, IRFunction):
            self.visit_function(ir)
        elif isinstance(ir, CFG):
            self.visit_cfg(ir)
        elif isinstance(ir, IRBasicBlock):
            self.visit_block(ir)
        return "\n".join(self._lines)

    def visit_module(self, module: IRModule) -> Any:
        self._emit(f"module {module.name} [{module.source_language.value}]")
        self._emit(f"source: {module.source_file}")
        self._emit("")

        if module.imports:
            self._emit("# imports")
            for imp in module.imports:
                self._emit(str(imp))
            self._emit("")

        if module.globals:
            self._emit("# globals")
            for gname, gvar in module.globals.items():
                self._emit(f"global {gvar}")
            self._emit("")

        for cls in module.classes.values():
            self.visit_class(cls)
            self._emit("")

        for func in module.functions.values():
            self.visit_function(func)
            self._emit("")

    def visit_class(self, cls: IRClass) -> Any:
        bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        for dec in cls.decorators:
            self._emit(f"@{dec}")
        self._emit(f"class {cls.name}{bases_str}:")
        self._indent_inc()
        if cls.mro:
            self._emit(f"# MRO: {' -> '.join(cls.mro)}")
        if cls.fields:
            for fname, ftype in cls.fields.items():
                self._emit(f"{fname}: {ftype}")
            self._emit("")
        for method in cls.methods.values():
            self.visit_function(method)
            self._emit("")
        self._indent_dec()

    def visit_function(self, func: IRFunction) -> Any:
        for dec in func.decorators:
            self._emit(f"@{dec}")
        params_str = ", ".join(str(p) for p in func.params)
        ret = f" -> {func.return_type}" if func.return_type else ""
        prefix = ""
        if func.is_async:
            prefix += "async "
        if func.is_generator:
            prefix += "gen "
        self._emit(f"def {prefix}{func.name}({params_str}){ret}:")
        if func.closure_captures:
            self._indent_inc()
            caps = ", ".join(str(c) for c in func.closure_captures)
            self._emit(f"# captures: {caps}")
            self._indent_dec()
        if func.body:
            self._indent_inc()
            self.visit_cfg(func.body)
            self._indent_dec()

    def visit_cfg(self, cfg: CFG) -> Any:
        rpo = cfg.reverse_postorder()
        for lbl in rpo:
            if lbl in cfg.blocks:
                self.visit_block(cfg.blocks[lbl])

    def visit_block(self, block: IRBasicBlock) -> Any:
        preds = ", ".join(block.predecessors) if block.predecessors else "none"
        self._emit(f"{block.label}:  # preds: {preds}")
        self._indent_inc()
        if block.idom is not None:
            self._emit(f"# idom: {block.idom}")
        if block.dominance_frontier:
            self._emit(f"# DF: {{{', '.join(sorted(block.dominance_frontier))}}}")
        for node in block.nodes:
            loc = f"  @ {node.location}" if node.location else ""
            self._emit(f"{node}{loc}")
        if block.terminator:
            loc = f"  @ {block.terminator.location}" if block.terminator.location else ""
            self._emit(f"{block.terminator}{loc}")
        self._indent_dec()


# ---------------------------------------------------------------------------
# IR Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """A single validation error."""
    message: str
    location: Optional[SourceLocation] = None
    block_label: Optional[str] = None
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        parts = [f"[{self.severity}]"]
        if self.block_label:
            parts.append(f"block {self.block_label}:")
        parts.append(self.message)
        if self.location:
            parts.append(f"at {self.location}")
        return " ".join(parts)


class IRValidator:
    """
    Validates SSA properties:
    - Each variable defined exactly once
    - Every use is dominated by its definition
    - Phi node correctness (one operand per predecessor)
    - CFG structural integrity
    """

    def __init__(self) -> None:
        self.errors: List[ValidationError] = []

    def validate_module(self, module: IRModule) -> List[ValidationError]:
        self.errors.clear()
        for func in module.functions.values():
            self.validate_function(func)
        for cls in module.classes.values():
            for method in cls.all_methods().values():
                self.validate_function(method)
        return self.errors

    def validate_function(self, func: IRFunction) -> List[ValidationError]:
        if func.body is None:
            return self.errors
        self.validate_cfg(func.body, func.name)
        return self.errors

    def validate_cfg(self, cfg: CFG, context: str = "") -> List[ValidationError]:
        self._validate_structure(cfg, context)
        self._validate_ssa(cfg, context)
        self._validate_phi_nodes(cfg, context)
        return self.errors

    def _validate_structure(self, cfg: CFG, context: str) -> None:
        """Validate CFG structural integrity."""
        # Entry block must exist
        if cfg.entry_label not in cfg.blocks:
            self.errors.append(ValidationError(
                message=f"Entry block '{cfg.entry_label}' not found in {context}",
            ))

        # Check edge consistency
        for lbl, blk in cfg.blocks.items():
            for succ in blk.successors:
                if succ not in cfg.blocks:
                    self.errors.append(ValidationError(
                        message=f"Successor '{succ}' of block '{lbl}' not found",
                        block_label=lbl,
                    ))
                elif lbl not in cfg.blocks[succ].predecessors:
                    self.errors.append(ValidationError(
                        message=f"Block '{lbl}' is successor of itself but not "
                                f"listed as predecessor in '{succ}'",
                        block_label=lbl,
                        severity="warning",
                    ))
            for pred in blk.predecessors:
                if pred not in cfg.blocks:
                    self.errors.append(ValidationError(
                        message=f"Predecessor '{pred}' of block '{lbl}' not found",
                        block_label=lbl,
                    ))

    def _validate_ssa(self, cfg: CFG, context: str) -> None:
        """Validate SSA property: each variable defined exactly once."""
        definitions: Dict[str, Tuple[str, IRNode]] = {}  # versioned_name -> (block, node)

        for lbl, blk in cfg.blocks.items():
            for inst in blk.all_instructions():
                for val in inst.defined_values():
                    vname = val.versioned_name
                    if vname in definitions:
                        prev_block, _prev_node = definitions[vname]
                        self.errors.append(ValidationError(
                            message=f"SSA violation: '{vname}' defined in both "
                                    f"'{prev_block}' and '{lbl}' in {context}",
                            block_label=lbl,
                            location=inst.location,
                        ))
                    else:
                        definitions[vname] = (lbl, inst)

        # Validate dominance: every use must be dominated by its definition
        dom_tree = DominatorTree(cfg)
        dom_tree.compute()

        def_blocks: Dict[str, str] = {vn: blk for vn, (blk, _) in definitions.items()}

        for lbl, blk in cfg.blocks.items():
            for inst in blk.all_instructions():
                if isinstance(inst, (PhiNode, GatedPhiNode)):
                    # Phi uses come from predecessors — check each incoming
                    for pred_lbl, val in inst.incoming.items():
                        vname = val.versioned_name
                        if vname in def_blocks:
                            def_blk = def_blocks[vname]
                            if not dom_tree.dominates(def_blk, pred_lbl):
                                self.errors.append(ValidationError(
                                    message=f"Dominance violation: '{vname}' used in "
                                            f"phi at '{lbl}' (from '{pred_lbl}') but "
                                            f"defined in '{def_blk}' which does not "
                                            f"dominate '{pred_lbl}' in {context}",
                                    block_label=lbl,
                                ))
                else:
                    for val in inst.used_values():
                        vname = val.versioned_name
                        if vname in def_blocks:
                            def_blk = def_blocks[vname]
                            if not dom_tree.dominates(def_blk, lbl):
                                self.errors.append(ValidationError(
                                    message=f"Dominance violation: '{vname}' used in "
                                            f"'{lbl}' but defined in '{def_blk}' which "
                                            f"does not dominate '{lbl}' in {context}",
                                    block_label=lbl,
                                    location=inst.location,
                                ))

    def _validate_phi_nodes(self, cfg: CFG, context: str) -> None:
        """Validate phi node correctness."""
        for lbl, blk in cfg.blocks.items():
            for node in blk.phi_nodes:
                if isinstance(node, PhiNode):
                    # Check one operand per predecessor
                    for pred in blk.predecessors:
                        if pred not in node.incoming:
                            self.errors.append(ValidationError(
                                message=f"Phi node for '{node.target}' in '{lbl}' "
                                        f"missing operand for predecessor '{pred}' "
                                        f"in {context}",
                                block_label=lbl,
                            ))
                    for inc_lbl in node.incoming:
                        if inc_lbl not in blk.predecessors:
                            self.errors.append(ValidationError(
                                message=f"Phi node for '{node.target}' in '{lbl}' "
                                        f"has operand from '{inc_lbl}' which is not "
                                        f"a predecessor in {context}",
                                block_label=lbl,
                            ))

    def is_valid(self) -> bool:
        return not any(e.severity == "error" for e in self.errors)

    def report(self) -> str:
        if not self.errors:
            return "IR validation passed."
        return "\n".join(str(e) for e in self.errors)


# ---------------------------------------------------------------------------
# IR Serializer — JSON serialization / deserialization
# ---------------------------------------------------------------------------

class IRSerializer:
    """Serialize and deserialize IR to/from JSON."""

    # --- Serialization ---

    def serialize_module(self, module: IRModule) -> Dict[str, Any]:
        return {
            "kind": "module",
            "name": module.name,
            "source_language": module.source_language.value,
            "source_file": module.source_file,
            "functions": {
                name: self.serialize_function(func)
                for name, func in module.functions.items()
            },
            "classes": {
                name: self.serialize_class(cls)
                for name, cls in module.classes.items()
            },
            "globals": {
                name: self._serialize_ssa_variable(var)
                for name, var in module.globals.items()
            },
            "imports": [self._serialize_node(imp) for imp in module.imports],
            "metadata": module.metadata,
        }

    def serialize_function(self, func: IRFunction) -> Dict[str, Any]:
        return {
            "kind": "function",
            "name": func.name,
            "params": [self._serialize_ssa_variable(p) for p in func.params],
            "return_type": self._serialize_type(func.return_type),
            "body": self.serialize_cfg(func.body) if func.body else None,
            "closure_captures": [
                self._serialize_ssa_variable(c) for c in func.closure_captures
            ],
            "is_async": func.is_async,
            "is_generator": func.is_generator,
            "decorators": func.decorators,
            "type_params": func.type_params,
            "location": self._serialize_location(func.location),
            "language_tag": func.language_tag.value if func.language_tag else None,
            "metadata": func.metadata,
        }

    def serialize_class(self, cls: IRClass) -> Dict[str, Any]:
        return {
            "kind": "class",
            "name": cls.name,
            "bases": cls.bases,
            "methods": {
                n: self.serialize_function(m) for n, m in cls.methods.items()
            },
            "static_methods": {
                n: self.serialize_function(m) for n, m in cls.static_methods.items()
            },
            "class_methods": {
                n: self.serialize_function(m) for n, m in cls.class_methods.items()
            },
            "fields": {n: self._serialize_type(t) for n, t in cls.fields.items()},
            "mro": cls.mro,
            "is_abstract": cls.is_abstract,
            "decorators": cls.decorators,
            "type_params": cls.type_params,
            "location": self._serialize_location(cls.location),
            "language_tag": cls.language_tag.value if cls.language_tag else None,
            "metadata": cls.metadata,
        }

    def serialize_cfg(self, cfg: CFG) -> Dict[str, Any]:
        return {
            "entry_label": cfg.entry_label,
            "exit_label": cfg.exit_label,
            "blocks": {
                lbl: self._serialize_block(blk)
                for lbl, blk in cfg.blocks.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "kind": e.kind.name,
                }
                for e in cfg.edges
            ],
        }

    def _serialize_block(self, block: IRBasicBlock) -> Dict[str, Any]:
        return {
            "label": block.label,
            "nodes": [self._serialize_node(n) for n in block.nodes],
            "terminator": self._serialize_node(block.terminator) if block.terminator else None,
            "predecessors": block.predecessors,
            "successors": block.successors,
        }

    def _serialize_node(self, node: IRNode) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": type(node).__name__,
            "id": node._id,
            "location": self._serialize_location(node.location),
            "language_tag": node.language_tag.value if node.language_tag else None,
        }

        if isinstance(node, PhiNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["incoming"] = {
                lbl: self._serialize_ssa_value(v)
                for lbl, v in node.incoming.items()
            }
        elif isinstance(node, GatedPhiNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["gate_kind"] = node.gate_kind
            data["incoming"] = {
                lbl: self._serialize_ssa_value(v)
                for lbl, v in node.incoming.items()
            }
            data["gate_condition"] = self._serialize_ssa_value(node.gate_condition)
        elif isinstance(node, TruthinessCoercionNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["operand"] = self._serialize_ssa_value(node.operand)
        elif isinstance(node, TypeNarrowNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["source"] = self._serialize_ssa_value(node.source)
            data["narrowed_type"] = self._serialize_type(node.narrowed_type)
            data["guard_condition"] = node.guard_condition
        elif isinstance(node, AssignNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["value"] = self._serialize_ssa_value(node.value)
        elif isinstance(node, ConstantNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["value"] = repr(node.value)
            data["type_annotation"] = self._serialize_type(node.type_annotation)
        elif isinstance(node, LiteralNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["literal_value"] = repr(node.literal_value)
            data["literal_type"] = self._serialize_type(node.literal_type)
        elif isinstance(node, BinOpNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["left"] = self._serialize_ssa_value(node.left)
            data["right"] = self._serialize_ssa_value(node.right)
            data["op"] = node.op.value
        elif isinstance(node, UnaryOpNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["operand"] = self._serialize_ssa_value(node.operand)
            data["op"] = node.op.value
        elif isinstance(node, CompareNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["left"] = self._serialize_ssa_value(node.left)
            data["right"] = self._serialize_ssa_value(node.right)
            data["op"] = node.op.value
        elif isinstance(node, CallNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["callee"] = self._serialize_ssa_value(node.callee)
            data["args"] = [self._serialize_ssa_value(a) for a in node.args]
            data["kwargs"] = {
                k: self._serialize_ssa_value(v) for k, v in node.kwargs.items()
            }
            data["is_new"] = node.is_new
        elif isinstance(node, LoadAttrNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["obj"] = self._serialize_ssa_value(node.obj)
            data["attr"] = node.attr
        elif isinstance(node, StoreAttrNode):
            data["obj"] = self._serialize_ssa_value(node.obj)
            data["attr"] = node.attr
            data["value"] = self._serialize_ssa_value(node.value)
        elif isinstance(node, LoadSubscriptNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["obj"] = self._serialize_ssa_value(node.obj)
            data["index"] = self._serialize_ssa_value(node.index)
        elif isinstance(node, StoreSubscriptNode):
            data["obj"] = self._serialize_ssa_value(node.obj)
            data["index"] = self._serialize_ssa_value(node.index)
            data["value"] = self._serialize_ssa_value(node.value)
        elif isinstance(node, ReturnNode):
            data["value"] = self._serialize_ssa_value(node.value)
        elif isinstance(node, BranchNode):
            data["target_label"] = node.target_label
        elif isinstance(node, ConditionalBranchNode):
            data["condition"] = self._serialize_ssa_value(node.condition)
            data["true_label"] = node.true_label
            data["false_label"] = node.false_label
        elif isinstance(node, YieldNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["value"] = self._serialize_ssa_value(node.value)
            data["is_yield_from"] = node.is_yield_from
            data["is_await"] = node.is_await
        elif isinstance(node, RaiseNode):
            data["exception"] = self._serialize_ssa_value(node.exception)
            data["cause"] = self._serialize_ssa_value(node.cause)
        elif isinstance(node, ImportNode):
            data["target"] = self._serialize_ssa_value(node.target)
            data["module_name"] = node.module_name
            data["imported_name"] = node.imported_name
            data["alias"] = node.alias

        return data

    def _serialize_ssa_value(self, val: Optional[SSAValue]) -> Optional[Dict[str, Any]]:
        if val is None:
            return None
        return {
            "name": val.name,
            "version": val.version,
            "type_annotation": self._serialize_type(val.type_annotation),
        }

    def _serialize_ssa_variable(self, var: SSAVariable) -> Dict[str, Any]:
        return {
            "ssa_value": self._serialize_ssa_value(var.ssa_value),
            "is_parameter": var.is_parameter,
            "is_closure_capture": var.is_closure_capture,
            "is_global": var.is_global,
        }

    def _serialize_type(self, t: Optional[IRType]) -> Optional[Dict[str, Any]]:
        if t is None:
            return None
        data: Dict[str, Any] = {
            "name": t.name,
            "args": [self._serialize_type(a) for a in t.args],
            "nullable": t.nullable,
        }
        if isinstance(t, RefinementType):
            data["kind"] = "refinement"
            data["base_type"] = self._serialize_type(t.base_type)
            data["predicate"] = t.predicate
        else:
            data["kind"] = "base"
        return data

    def _serialize_location(self, loc: Optional[SourceLocation]) -> Optional[Dict[str, Any]]:
        if loc is None:
            return None
        return {
            "file": loc.file,
            "line": loc.line,
            "column": loc.column,
            "end_line": loc.end_line,
            "end_column": loc.end_column,
        }

    def to_json(self, module: IRModule, indent: int = 2) -> str:
        return json.dumps(self.serialize_module(module), indent=indent)

    # --- Deserialization ---

    def deserialize_module(self, data: Dict[str, Any]) -> IRModule:
        module = IRModule(
            name=data.get("name", "<module>"),
            source_language=LanguageTag(data.get("source_language", "python")),
            source_file=data.get("source_file", "<unknown>"),
            metadata=data.get("metadata", {}),
        )

        for name, fdata in data.get("functions", {}).items():
            module.functions[name] = self.deserialize_function(fdata)

        for name, cdata in data.get("classes", {}).items():
            module.classes[name] = self.deserialize_class(cdata)

        for name, vdata in data.get("globals", {}).items():
            module.globals[name] = self._deserialize_ssa_variable(vdata)

        for idata in data.get("imports", []):
            module.imports.append(self._deserialize_import_node(idata))

        return module

    def deserialize_function(self, data: Dict[str, Any]) -> IRFunction:
        func = IRFunction(
            name=data.get("name", ""),
            params=[
                self._deserialize_ssa_variable(p) for p in data.get("params", [])
            ],
            return_type=self._deserialize_type(data.get("return_type")),
            is_async=data.get("is_async", False),
            is_generator=data.get("is_generator", False),
            decorators=data.get("decorators", []),
            type_params=data.get("type_params", []),
            location=self._deserialize_location(data.get("location")),
            language_tag=LanguageTag(data["language_tag"]) if data.get("language_tag") else None,
            metadata=data.get("metadata", {}),
        )
        func.closure_captures = [
            self._deserialize_ssa_variable(c)
            for c in data.get("closure_captures", [])
        ]
        body_data = data.get("body")
        if body_data:
            func.body = self.deserialize_cfg(body_data)
        return func

    def deserialize_class(self, data: Dict[str, Any]) -> IRClass:
        cls = IRClass(
            name=data.get("name", ""),
            bases=data.get("bases", []),
            mro=data.get("mro", []),
            is_abstract=data.get("is_abstract", False),
            decorators=data.get("decorators", []),
            type_params=data.get("type_params", []),
            location=self._deserialize_location(data.get("location")),
            language_tag=LanguageTag(data["language_tag"]) if data.get("language_tag") else None,
            metadata=data.get("metadata", {}),
        )
        for n, t in data.get("fields", {}).items():
            cls.fields[n] = self._deserialize_type(t) or UNKNOWN_TYPE
        for n, m in data.get("methods", {}).items():
            cls.methods[n] = self.deserialize_function(m)
        for n, m in data.get("static_methods", {}).items():
            cls.static_methods[n] = self.deserialize_function(m)
        for n, m in data.get("class_methods", {}).items():
            cls.class_methods[n] = self.deserialize_function(m)
        return cls

    def deserialize_cfg(self, data: Dict[str, Any]) -> CFG:
        cfg = CFG(
            entry_label=data.get("entry_label", "entry"),
            exit_label=data.get("exit_label", "exit"),
        )
        for lbl, bdata in data.get("blocks", {}).items():
            cfg.blocks[lbl] = self._deserialize_block(bdata)
        for edata in data.get("edges", []):
            kind = EdgeKind[edata.get("kind", "Normal")]
            cfg.add_edge(edata["source"], edata["target"], kind)
        return cfg

    def _deserialize_block(self, data: Dict[str, Any]) -> IRBasicBlock:
        block = IRBasicBlock(
            label=data.get("label", ""),
            predecessors=data.get("predecessors", []),
            successors=data.get("successors", []),
        )
        for ndata in data.get("nodes", []):
            block.nodes.append(self._deserialize_node(ndata))
        term_data = data.get("terminator")
        if term_data:
            block.terminator = self._deserialize_node(term_data)
        return block

    def _deserialize_node(self, data: Dict[str, Any]) -> IRNode:
        """Deserialize a node from JSON data."""
        node_type = data.get("type", "IRNode")
        location = self._deserialize_location(data.get("location"))
        lang_tag = LanguageTag(data["language_tag"]) if data.get("language_tag") else None

        if node_type == "PhiNode":
            incoming = {}
            for lbl, vdata in data.get("incoming", {}).items():
                if vdata:
                    incoming[lbl] = self._deserialize_ssa_value(vdata)
            return PhiNode(
                target=self._deserialize_ssa_value(data.get("target")),
                incoming=incoming,
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "GatedPhiNode":
            incoming = {}
            for lbl, vdata in data.get("incoming", {}).items():
                if vdata:
                    incoming[lbl] = self._deserialize_ssa_value(vdata)
            return GatedPhiNode(
                target=self._deserialize_ssa_value(data.get("target")),
                gate_kind=data.get("gate_kind", "exception"),
                incoming=incoming,
                gate_condition=self._deserialize_ssa_value(data.get("gate_condition")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "TruthinessCoercionNode":
            return TruthinessCoercionNode(
                target=self._deserialize_ssa_value(data.get("target")),
                operand=self._deserialize_ssa_value(data.get("operand")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "TypeNarrowNode":
            return TypeNarrowNode(
                target=self._deserialize_ssa_value(data.get("target")),
                source=self._deserialize_ssa_value(data.get("source")),
                narrowed_type=self._deserialize_type(data.get("narrowed_type")),
                guard_condition=data.get("guard_condition"),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "AssignNode":
            return AssignNode(
                target=self._deserialize_ssa_value(data.get("target")),
                value=self._deserialize_ssa_value(data.get("value")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "ConstantNode":
            return ConstantNode(
                target=self._deserialize_ssa_value(data.get("target")),
                value=data.get("value"),
                type_annotation=self._deserialize_type(data.get("type_annotation")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "LiteralNode":
            return LiteralNode(
                target=self._deserialize_ssa_value(data.get("target")),
                literal_value=data.get("literal_value"),
                literal_type=self._deserialize_type(data.get("literal_type")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "BinOpNode":
            return BinOpNode(
                target=self._deserialize_ssa_value(data.get("target")),
                left=self._deserialize_ssa_value(data.get("left")),
                right=self._deserialize_ssa_value(data.get("right")),
                op=BinOp(data.get("op", "+")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "UnaryOpNode":
            return UnaryOpNode(
                target=self._deserialize_ssa_value(data.get("target")),
                operand=self._deserialize_ssa_value(data.get("operand")),
                op=UnaryOp(data.get("op", "-")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "CompareNode":
            return CompareNode(
                target=self._deserialize_ssa_value(data.get("target")),
                left=self._deserialize_ssa_value(data.get("left")),
                right=self._deserialize_ssa_value(data.get("right")),
                op=CompareOp(data.get("op", "==")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "CallNode":
            return CallNode(
                target=self._deserialize_ssa_value(data.get("target")),
                callee=self._deserialize_ssa_value(data.get("callee")),
                args=[self._deserialize_ssa_value(a) for a in data.get("args", []) if a],
                kwargs={
                    k: self._deserialize_ssa_value(v)
                    for k, v in data.get("kwargs", {}).items()
                    if v
                },
                is_new=data.get("is_new", False),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "LoadAttrNode":
            return LoadAttrNode(
                target=self._deserialize_ssa_value(data.get("target")),
                obj=self._deserialize_ssa_value(data.get("obj")),
                attr=data.get("attr", ""),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "StoreAttrNode":
            return StoreAttrNode(
                obj=self._deserialize_ssa_value(data.get("obj")),
                attr=data.get("attr", ""),
                value=self._deserialize_ssa_value(data.get("value")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "LoadSubscriptNode":
            return LoadSubscriptNode(
                target=self._deserialize_ssa_value(data.get("target")),
                obj=self._deserialize_ssa_value(data.get("obj")),
                index=self._deserialize_ssa_value(data.get("index")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "StoreSubscriptNode":
            return StoreSubscriptNode(
                obj=self._deserialize_ssa_value(data.get("obj")),
                index=self._deserialize_ssa_value(data.get("index")),
                value=self._deserialize_ssa_value(data.get("value")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "ReturnNode":
            return ReturnNode(
                value=self._deserialize_ssa_value(data.get("value")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "BranchNode":
            return BranchNode(
                target_label=data.get("target_label", ""),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "ConditionalBranchNode":
            return ConditionalBranchNode(
                condition=self._deserialize_ssa_value(data.get("condition")),
                true_label=data.get("true_label", ""),
                false_label=data.get("false_label", ""),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "YieldNode":
            return YieldNode(
                target=self._deserialize_ssa_value(data.get("target")),
                value=self._deserialize_ssa_value(data.get("value")),
                is_yield_from=data.get("is_yield_from", False),
                is_await=data.get("is_await", False),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "RaiseNode":
            return RaiseNode(
                exception=self._deserialize_ssa_value(data.get("exception")),
                cause=self._deserialize_ssa_value(data.get("cause")),
                location=location,
                language_tag=lang_tag,
            )
        elif node_type == "ImportNode":
            return self._deserialize_import_node(data)

        return IRNode(location=location, language_tag=lang_tag)

    def _deserialize_import_node(self, data: Dict[str, Any]) -> ImportNode:
        return ImportNode(
            target=self._deserialize_ssa_value(data.get("target")),
            module_name=data.get("module_name", ""),
            imported_name=data.get("imported_name"),
            alias=data.get("alias"),
            location=self._deserialize_location(data.get("location")),
            language_tag=LanguageTag(data["language_tag"]) if data.get("language_tag") else None,
        )

    def _deserialize_ssa_value(self, data: Optional[Dict[str, Any]]) -> Optional[SSAValue]:
        if data is None:
            return None
        return SSAValue(
            name=data.get("name", ""),
            version=data.get("version", 0),
            type_annotation=self._deserialize_type(data.get("type_annotation")),
        )

    def _deserialize_ssa_variable(self, data: Dict[str, Any]) -> SSAVariable:
        return SSAVariable(
            ssa_value=self._deserialize_ssa_value(data.get("ssa_value")) or SSAValue(""),
            is_parameter=data.get("is_parameter", False),
            is_closure_capture=data.get("is_closure_capture", False),
            is_global=data.get("is_global", False),
        )

    def _deserialize_type(self, data: Optional[Dict[str, Any]]) -> Optional[IRType]:
        if data is None:
            return None
        if data.get("kind") == "refinement":
            return RefinementType(
                name=data.get("name", ""),
                args=[self._deserialize_type(a) or UNKNOWN_TYPE for a in data.get("args", [])],
                nullable=data.get("nullable", False),
                base_type=self._deserialize_type(data.get("base_type")),
                predicate=data.get("predicate"),
            )
        return IRType(
            name=data.get("name", ""),
            args=[self._deserialize_type(a) or UNKNOWN_TYPE for a in data.get("args", [])],
            nullable=data.get("nullable", False),
        )

    def _deserialize_location(
        self, data: Optional[Dict[str, Any]]
    ) -> Optional[SourceLocation]:
        if data is None:
            return None
        return SourceLocation(
            file=data.get("file", "<unknown>"),
            line=data.get("line", 0),
            column=data.get("column", 0),
            end_line=data.get("end_line"),
            end_column=data.get("end_column"),
        )

    def from_json(self, text: str) -> IRModule:
        return self.deserialize_module(json.loads(text))


# ---------------------------------------------------------------------------
# Self-test: build a small IR, run SSA construction, validate, print, serialize
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke test: build IR for a simple function with a branch."""
    IRNode.reset_id_counter()

    # Build: def f(x):
    #            if x > 0:
    #                y = x + 1
    #            else:
    #                y = x - 1
    #            return y

    builder = CFGBuilder()
    entry = builder.new_block("entry")
    then_blk = builder.new_block("then")
    else_blk = builder.new_block("else")
    merge = builder.new_block("merge")
    exit_blk = builder.new_block("exit")

    x = SSAValue("x", 0, INT_TYPE)
    zero = SSAValue("_const_0", 0)
    cond = SSAValue("cond", 0, BOOL_TYPE)
    y_then = SSAValue("y", 0, INT_TYPE)
    y_else = SSAValue("y", 0, INT_TYPE)
    one = SSAValue("_const_1", 0, INT_TYPE)
    y_merge = SSAValue("y", 0, INT_TYPE)

    # Entry
    builder.set_current_block("entry")
    builder.emit(ConstantNode(target=zero, value=0, type_annotation=INT_TYPE))
    builder.emit(CompareNode(target=cond, left=x, right=zero, op=CompareOp.Gt))
    builder.set_terminator(ConditionalBranchNode(
        condition=cond, true_label="then", false_label="else"
    ))

    # Then
    builder.set_current_block("then")
    builder.emit(ConstantNode(target=SSAValue("_const_1", 0), value=1, type_annotation=INT_TYPE))
    builder.emit(BinOpNode(target=y_then, left=x, right=SSAValue("_const_1", 0), op=BinOp.Add))
    builder.set_terminator(BranchNode(target_label="merge"))

    # Else
    builder.set_current_block("else")
    builder.emit(ConstantNode(target=SSAValue("_const_1", 0), value=1, type_annotation=INT_TYPE))
    builder.emit(BinOpNode(target=y_else, left=x, right=SSAValue("_const_1", 0), op=BinOp.Sub))
    builder.set_terminator(BranchNode(target_label="merge"))

    # Merge — phi will be inserted by SSA builder
    builder.set_current_block("merge")
    builder.emit(AssignNode(target=y_merge, value=SSAValue("y", 0)))
    builder.set_terminator(BranchNode(target_label="exit"))

    # Exit
    builder.set_current_block("exit")
    builder.set_terminator(ReturnNode(value=SSAValue("y", 0)))

    builder.add_edge("entry", "then", EdgeKind.TrueGuard)
    builder.add_edge("entry", "else", EdgeKind.FalseGuard)
    builder.add_edge("then", "merge")
    builder.add_edge("else", "merge")
    builder.add_edge("merge", "exit")

    cfg = builder.finalize()

    # Build SSA
    ssa = SSABuilder(cfg)
    ssa.build(initial_defs={"x": 1})

    # Build function and module
    func = IRFunction(
        name="f",
        params=[SSAVariable(SSAValue("x", 0, INT_TYPE), is_parameter=True)],
        return_type=INT_TYPE,
        body=cfg,
        language_tag=LanguageTag.Python,
    )

    module = IRModule(
        name="test_module",
        source_language=LanguageTag.Python,
        source_file="test.py",
    )
    module.add_function(func)

    # Print
    printer = IRPrinter()
    output = printer.print(module)
    assert "module test_module" in output, "Printer failed"

    # Validate
    validator = IRValidator()
    validator.validate_module(module)

    # Serialize round-trip
    serializer = IRSerializer()
    json_str = serializer.to_json(module)
    parsed = json.loads(json_str)
    assert parsed["name"] == "test_module"
    module2 = serializer.from_json(json_str)
    assert module2.name == "test_module"
    assert "f" in module2.functions

    # Dominator tree
    dom = DominatorTree(cfg)
    dom.compute()
    assert dom.dominates("entry", "merge"), "Dominance check failed"
    assert dom.dominates("entry", "then"), "Dominance check failed"

    # C3 MRO test
    base_cls = IRClass(name="Base", mro=["Base"])
    child_cls = IRClass(name="Child", bases=["Base"])
    registry: Dict[str, IRClass] = {"Base": base_cls, "Child": child_cls}
    child_cls.compute_mro(registry)
    assert child_cls.mro == ["Child", "Base"], f"MRO failed: {child_cls.mro}"

    print("unified.py self-test passed.")


if __name__ == "__main__":
    _self_test()
