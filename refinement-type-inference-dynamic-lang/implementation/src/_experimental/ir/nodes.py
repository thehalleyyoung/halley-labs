"""
IR node types for SSA-based intermediate representation used in
refinement type inference for dynamically-typed languages.

Each node carries source location, SSA def/use information, and
type annotations.  Nodes support visitor pattern, pretty-printing,
cloning, and SSA use-rewriting.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Source position attached to every IR node."""
    file: str = "<unknown>"
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


# ---------------------------------------------------------------------------
# SSA variable reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SSAVar:
    """A versioned SSA variable reference."""
    name: str
    version: int = 0

    def __str__(self) -> str:
        return f"{self.name}_{self.version}" if self.version else self.name

    def __repr__(self) -> str:
        return f"SSAVar({self.name!r}, {self.version})"


# ---------------------------------------------------------------------------
# Type annotation stub (filled in by the type inference pass)
# ---------------------------------------------------------------------------

@dataclass
class TypeAnnotation:
    """Placeholder for a refinement type assigned during inference."""
    raw: str = "unknown"
    refined: Optional[str] = None

    def __str__(self) -> str:
        return self.refined or self.raw


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BinOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOOR_DIV = auto()
    MOD = auto()
    POW = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    AND = auto()
    OR = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    IS = auto()
    IS_NOT = auto()
    IN = auto()
    NOT_IN = auto()
    MAT_MUL = auto()


class UnaryOp(Enum):
    NOT = auto()
    NEGATE = auto()
    INVERT = auto()
    TYPEOF = auto()
    IS_TRUTHY = auto()
    POS = auto()


class CompareOp(Enum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()
    IS = auto()
    IS_NOT = auto()
    IN = auto()
    NOT_IN = auto()


class GuardKind(Enum):
    ISINSTANCE = auto()
    TYPEOF = auto()
    IS_NONE = auto()
    IS_NOT_NONE = auto()
    HASATTR = auto()
    COMPARISON = auto()
    TRUTHINESS = auto()
    CALLABLE = auto()
    CUSTOM = auto()


class ContainerKind(Enum):
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    FROZENSET = auto()


class LiteralKind(Enum):
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BYTES = auto()
    BOOL = auto()
    NONE = auto()
    ELLIPSIS = auto()
    COMPLEX = auto()


# ---------------------------------------------------------------------------
# Visitor protocol
# ---------------------------------------------------------------------------

class IRVisitor(ABC):
    """Visitor interface for IR nodes.  Subclass and override ``visit_*``."""

    def visit(self, node: "IRNode") -> Any:
        return node.accept(self)

    # Default implementations simply return ``None``.
    def visit_assign(self, node: "AssignNode") -> Any: return None
    def visit_phi(self, node: "PhiNode") -> Any: return None
    def visit_gated_phi(self, node: "GatedPhiNode") -> Any: return None
    def visit_guard(self, node: "GuardNode") -> Any: return None
    def visit_call(self, node: "CallNode") -> Any: return None
    def visit_return(self, node: "ReturnNode") -> Any: return None
    def visit_branch(self, node: "BranchNode") -> Any: return None
    def visit_jump(self, node: "JumpNode") -> Any: return None
    def visit_binop(self, node: "BinOpNode") -> Any: return None
    def visit_unaryop(self, node: "UnaryOpNode") -> Any: return None
    def visit_compare(self, node: "CompareNode") -> Any: return None
    def visit_load_attr(self, node: "LoadAttrNode") -> Any: return None
    def visit_store_attr(self, node: "StoreAttrNode") -> Any: return None
    def visit_index(self, node: "IndexNode") -> Any: return None
    def visit_store_index(self, node: "StoreIndexNode") -> Any: return None
    def visit_literal(self, node: "LiteralNode") -> Any: return None
    def visit_truthiness(self, node: "TruthinessNode") -> Any: return None
    def visit_type_narrow(self, node: "TypeNarrowNode") -> Any: return None
    def visit_type_test(self, node: "TypeTestNode") -> Any: return None
    def visit_null_check(self, node: "NullCheckNode") -> Any: return None
    def visit_has_attr(self, node: "HasAttrNode") -> Any: return None
    def visit_len(self, node: "LenNode") -> Any: return None
    def visit_container_create(self, node: "ContainerCreateNode") -> Any: return None
    def visit_unpack(self, node: "UnpackNode") -> Any: return None
    def visit_yield(self, node: "YieldNode") -> Any: return None
    def visit_await(self, node: "AwaitNode") -> Any: return None
    def visit_raise(self, node: "RaiseNode") -> Any: return None
    def visit_except_handler(self, node: "ExceptHandlerNode") -> Any: return None
    def visit_import(self, node: "ImportNode") -> Any: return None
    def visit_delete(self, node: "DeleteNode") -> Any: return None
    def visit_assert(self, node: "AssertNode") -> Any: return None
    def visit_slice(self, node: "SliceNode") -> Any: return None
    def visit_format_string(self, node: "FormatStringNode") -> Any: return None
    def visit_closure_capture(self, node: "ClosureCaptureNode") -> Any: return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_next_node_id: int = 0


def _fresh_id() -> int:
    global _next_node_id
    nid = _next_node_id
    _next_node_id += 1
    return nid


def reset_node_ids(start: int = 0) -> None:
    """Reset the global node-id counter (useful in tests)."""
    global _next_node_id
    _next_node_id = start


def _replace_var(
    var: Optional[SSAVar],
    mapping: Mapping[SSAVar, SSAVar],
) -> Optional[SSAVar]:
    if var is None:
        return None
    return mapping.get(var, var)


def _replace_var_list(
    vars_: List[SSAVar],
    mapping: Mapping[SSAVar, SSAVar],
) -> List[SSAVar]:
    return [mapping.get(v, v) for v in vars_]


def _var_set(v: Optional[SSAVar]) -> FrozenSet[SSAVar]:
    return frozenset({v}) if v is not None else frozenset()


def _vars_from_list(vs: Sequence[Optional[SSAVar]]) -> FrozenSet[SSAVar]:
    return frozenset(v for v in vs if v is not None)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class IRNode(ABC):
    """Abstract base for every IR node."""

    node_id: int = field(default_factory=_fresh_id, init=False, repr=False)
    source_loc: SourceLocation = field(default_factory=SourceLocation)
    type_ann: TypeAnnotation = field(default_factory=TypeAnnotation, repr=False)

    # -- SSA interface -------------------------------------------------------

    @property
    @abstractmethod
    def defined_vars(self) -> FrozenSet[SSAVar]:
        """SSA variables defined (written) by this node."""
        ...

    @property
    @abstractmethod
    def used_vars(self) -> FrozenSet[SSAVar]:
        """SSA variables used (read) by this node."""
        ...

    @abstractmethod
    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        """Rewrite every *use* of an SSA variable according to *mapping*."""
        ...

    # -- Visitor / utility ---------------------------------------------------

    @abstractmethod
    def accept(self, visitor: IRVisitor) -> Any:
        ...

    @abstractmethod
    def pretty_print(self, indent: int = 0) -> str:
        ...

    def clone(self) -> "IRNode":
        """Deep-copy the node, assigning a fresh ``node_id``."""
        c = copy.deepcopy(self)
        c.node_id = _fresh_id()
        return c

    def __str__(self) -> str:
        return self.pretty_print()

    def _indent(self, indent: int) -> str:
        return "  " * indent


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------

@dataclass
class AssignNode(IRNode):
    """``dst = src``"""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    src: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.src})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.src = mapping.get(self.src, self.src)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_assign(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = {self.src}"


# ---------------------------------------------------------------------------

@dataclass
class PhiNode(IRNode):
    """SSA φ-function: ``dst = φ(incoming…)``

    Each entry in *incoming* maps a predecessor block label to an SSA
    variable.  *gating_condition* is an optional SSA var that indicates
    which predecessor was actually taken (for guard-/exception-aware φ).
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    incoming: Dict[str, SSAVar] = field(default_factory=dict)
    gating_condition: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = set(self.incoming.values())
        if self.gating_condition is not None:
            vs.add(self.gating_condition)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.incoming = {
            blk: mapping.get(v, v) for blk, v in self.incoming.items()
        }
        self.gating_condition = _replace_var(self.gating_condition, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_phi(self)

    def pretty_print(self, indent: int = 0) -> str:
        arms = ", ".join(f"{blk}: {v}" for blk, v in self.incoming.items())
        gate = f" gated_by {self.gating_condition}" if self.gating_condition else ""
        return f"{self._indent(indent)}{self.dst} = φ({arms}){gate}"


# ---------------------------------------------------------------------------

@dataclass
class GatedPhiNode(IRNode):
    """Phi with an explicit guard condition for type narrowing.

    After a guard like ``isinstance(x, int)`` the true branch narrows x
    to ``int`` and the false branch to ``~int``.  This node merges both
    versions with the guard result as selector.

    ``dst = gated_φ(guard_var, true_var, false_var)``
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    guard_var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    true_var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    false_var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    guard_kind: GuardKind = GuardKind.ISINSTANCE
    narrowed_type_true: Optional[str] = None
    narrowed_type_false: Optional[str] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.guard_var, self.true_var, self.false_var})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.guard_var = mapping.get(self.guard_var, self.guard_var)
        self.true_var = mapping.get(self.true_var, self.true_var)
        self.false_var = mapping.get(self.false_var, self.false_var)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_gated_phi(self)

    def pretty_print(self, indent: int = 0) -> str:
        tt = f" [{self.narrowed_type_true}]" if self.narrowed_type_true else ""
        ft = f" [{self.narrowed_type_false}]" if self.narrowed_type_false else ""
        return (
            f"{self._indent(indent)}{self.dst} = gated_φ("
            f"{self.guard_var}, true={self.true_var}{tt}, "
            f"false={self.false_var}{ft})"
        )


# ---------------------------------------------------------------------------

@dataclass
class GuardNode(IRNode):
    """Runtime type/value guard.

    Encodes checks like ``isinstance(x, T)``, ``x is None``,
    ``hasattr(x, 'k')``, or arbitrary comparisons.
    """
    guard_kind: GuardKind = GuardKind.ISINSTANCE
    subject: SSAVar = field(default_factory=lambda: SSAVar("_"))
    guard_args: List[Any] = field(default_factory=list)
    result_var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    true_target: str = ""
    false_target: str = ""
    narrowed_type_true: Optional[str] = None
    narrowed_type_false: Optional[str] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.result_var})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = {self.subject}
        for a in self.guard_args:
            if isinstance(a, SSAVar):
                vs.add(a)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.subject = mapping.get(self.subject, self.subject)
        self.guard_args = [
            mapping.get(a, a) if isinstance(a, SSAVar) else a
            for a in self.guard_args
        ]

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_guard(self)

    def pretty_print(self, indent: int = 0) -> str:
        args_s = ", ".join(str(a) for a in self.guard_args)
        return (
            f"{self._indent(indent)}{self.result_var} = guard<{self.guard_kind.name}>"
            f"({self.subject}, {args_s}) "
            f"-> T:{self.true_target} F:{self.false_target}"
        )


# ---------------------------------------------------------------------------

@dataclass
class CallNode(IRNode):
    """Function or method call.

    ``result = receiver.method(args…, **kwargs)``  (receiver may be None
    for plain function calls).
    """
    result: SSAVar = field(default_factory=lambda: SSAVar("_"))
    func: SSAVar = field(default_factory=lambda: SSAVar("_"))
    receiver: Optional[SSAVar] = None
    args: List[SSAVar] = field(default_factory=list)
    kwargs: Dict[str, SSAVar] = field(default_factory=dict)
    is_method: bool = False
    star_args: Optional[SSAVar] = None
    star_kwargs: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.result})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = {self.func}
        if self.receiver is not None:
            vs.add(self.receiver)
        vs.update(self.args)
        vs.update(self.kwargs.values())
        if self.star_args is not None:
            vs.add(self.star_args)
        if self.star_kwargs is not None:
            vs.add(self.star_kwargs)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.func = mapping.get(self.func, self.func)
        self.receiver = _replace_var(self.receiver, mapping)
        self.args = _replace_var_list(self.args, mapping)
        self.kwargs = {k: mapping.get(v, v) for k, v in self.kwargs.items()}
        self.star_args = _replace_var(self.star_args, mapping)
        self.star_kwargs = _replace_var(self.star_kwargs, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_call(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        rcv = f"{self.receiver}." if self.receiver else ""
        a = ", ".join(str(a) for a in self.args)
        kw = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        parts = [p for p in (a, kw) if p]
        if self.star_args:
            parts.append(f"*{self.star_args}")
        if self.star_kwargs:
            parts.append(f"**{self.star_kwargs}")
        return f"{pfx}{self.result} = {rcv}{self.func}({', '.join(parts)})"


# ---------------------------------------------------------------------------

@dataclass
class ReturnNode(IRNode):
    """``return value``"""
    value: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return _var_set(self.value)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.value = _replace_var(self.value, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_return(self)

    def pretty_print(self, indent: int = 0) -> str:
        v = f" {self.value}" if self.value else ""
        return f"{self._indent(indent)}return{v}"


# ---------------------------------------------------------------------------

@dataclass
class BranchNode(IRNode):
    """Conditional branch."""
    condition: SSAVar = field(default_factory=lambda: SSAVar("_"))
    true_block: str = ""
    false_block: str = ""

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.condition})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.condition = mapping.get(self.condition, self.condition)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_branch(self)

    def pretty_print(self, indent: int = 0) -> str:
        return (
            f"{self._indent(indent)}branch {self.condition} "
            f"-> T:{self.true_block} F:{self.false_block}"
        )


# ---------------------------------------------------------------------------

@dataclass
class JumpNode(IRNode):
    """Unconditional jump."""
    target: str = ""

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        pass

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_jump(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}jump -> {self.target}"


# ---------------------------------------------------------------------------

@dataclass
class BinOpNode(IRNode):
    """Binary operation: ``dst = left op right``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    op: BinOp = BinOp.ADD
    left: SSAVar = field(default_factory=lambda: SSAVar("_"))
    right: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.left, self.right})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.left = mapping.get(self.left, self.left)
        self.right = mapping.get(self.right, self.right)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_binop(self)

    def pretty_print(self, indent: int = 0) -> str:
        return (
            f"{self._indent(indent)}{self.dst} = "
            f"{self.left} {self.op.name} {self.right}"
        )


# ---------------------------------------------------------------------------

@dataclass
class UnaryOpNode(IRNode):
    """Unary operation: ``dst = op operand``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    op: UnaryOp = UnaryOp.NOT
    operand: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.operand})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.operand = mapping.get(self.operand, self.operand)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_unaryop(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = {self.op.name} {self.operand}"


# ---------------------------------------------------------------------------

@dataclass
class CompareNode(IRNode):
    """Chained comparison (Python ``a < b < c``).

    *operands* has len N, *ops* has len N-1.
    ``dst = operands[0] ops[0] operands[1] ops[1] …``
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    ops: List[CompareOp] = field(default_factory=list)
    operands: List[SSAVar] = field(default_factory=list)

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset(self.operands)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.operands = _replace_var_list(self.operands, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_compare(self)

    def pretty_print(self, indent: int = 0) -> str:
        parts: List[str] = []
        for i, operand in enumerate(self.operands):
            parts.append(str(operand))
            if i < len(self.ops):
                parts.append(self.ops[i].name)
        return f"{self._indent(indent)}{self.dst} = {' '.join(parts)}"


# ---------------------------------------------------------------------------

@dataclass
class LoadAttrNode(IRNode):
    """Attribute load: ``dst = obj.attr``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    attr: str = ""

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_load_attr(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = {self.obj}.{self.attr}"


# ---------------------------------------------------------------------------

@dataclass
class StoreAttrNode(IRNode):
    """Attribute store: ``obj.attr = value``."""
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    attr: str = ""
    value: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj, self.value})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)
        self.value = mapping.get(self.value, self.value)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_store_attr(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.obj}.{self.attr} = {self.value}"


# ---------------------------------------------------------------------------

@dataclass
class IndexNode(IRNode):
    """Subscript load: ``dst = obj[index]``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    index: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj, self.index})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)
        self.index = mapping.get(self.index, self.index)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_index(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = {self.obj}[{self.index}]"


# ---------------------------------------------------------------------------

@dataclass
class StoreIndexNode(IRNode):
    """Subscript store: ``obj[index] = value``."""
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    index: SSAVar = field(default_factory=lambda: SSAVar("_"))
    value: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj, self.index, self.value})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)
        self.index = mapping.get(self.index, self.index)
        self.value = mapping.get(self.value, self.value)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_store_index(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.obj}[{self.index}] = {self.value}"


# ---------------------------------------------------------------------------

@dataclass
class LiteralNode(IRNode):
    """Literal value: ``dst = <literal>``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    kind: LiteralKind = LiteralKind.NONE
    value: Any = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        pass

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_literal(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = {self.kind.name}({self.value!r})"


# ---------------------------------------------------------------------------

@dataclass
class TruthinessNode(IRNode):
    """Explicit truthiness coercion: ``dst = bool(operand)``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    operand: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.operand})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.operand = mapping.get(self.operand, self.operand)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_truthiness(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = bool({self.operand})"


# ---------------------------------------------------------------------------

@dataclass
class TypeNarrowNode(IRNode):
    """Explicit type narrowing at a guard boundary.

    ``dst = narrow(src, narrowed_type)``  — inserted at the entry of a
    guard's true/false successor to record the refined type.
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    src: SSAVar = field(default_factory=lambda: SSAVar("_"))
    narrowed_type: str = "unknown"
    guard_node_id: Optional[int] = None
    is_true_branch: bool = True

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.src})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.src = mapping.get(self.src, self.src)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_type_narrow(self)

    def pretty_print(self, indent: int = 0) -> str:
        branch = "T" if self.is_true_branch else "F"
        return (
            f"{self._indent(indent)}{self.dst} = narrow<{self.narrowed_type}, "
            f"{branch}>({self.src})"
        )


# ---------------------------------------------------------------------------

@dataclass
class TypeTestNode(IRNode):
    """``isinstance`` / ``typeof`` test: ``dst = isinstance(subject, tested_type)``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    subject: SSAVar = field(default_factory=lambda: SSAVar("_"))
    tested_type: str = ""
    is_typeof: bool = False

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.subject})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.subject = mapping.get(self.subject, self.subject)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_type_test(self)

    def pretty_print(self, indent: int = 0) -> str:
        fn = "typeof" if self.is_typeof else "isinstance"
        return (
            f"{self._indent(indent)}{self.dst} = "
            f"{fn}({self.subject}, {self.tested_type})"
        )


# ---------------------------------------------------------------------------

@dataclass
class NullCheckNode(IRNode):
    """``is None`` / ``=== null`` test: ``dst = (subject is None)``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    subject: SSAVar = field(default_factory=lambda: SSAVar("_"))
    negated: bool = False

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.subject})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.subject = mapping.get(self.subject, self.subject)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_null_check(self)

    def pretty_print(self, indent: int = 0) -> str:
        op = "is not" if self.negated else "is"
        return f"{self._indent(indent)}{self.dst} = ({self.subject} {op} None)"


# ---------------------------------------------------------------------------

@dataclass
class HasAttrNode(IRNode):
    """``hasattr(obj, attr_name)`` / ``'k' in obj``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    attr_name: str = ""

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_has_attr(self)

    def pretty_print(self, indent: int = 0) -> str:
        return (
            f"{self._indent(indent)}{self.dst} = "
            f"hasattr({self.obj}, {self.attr_name!r})"
        )


# ---------------------------------------------------------------------------

@dataclass
class LenNode(IRNode):
    """``dst = len(obj)``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.obj})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_len(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = len({self.obj})"


# ---------------------------------------------------------------------------

@dataclass
class ContainerCreateNode(IRNode):
    """Container creation: ``dst = [e0, e1, …]`` / ``{k: v, …}`` / etc."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    kind: ContainerKind = ContainerKind.LIST
    elements: List[SSAVar] = field(default_factory=list)
    keys: List[Optional[SSAVar]] = field(default_factory=list)

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = set(self.elements)
        for k in self.keys:
            if k is not None:
                vs.add(k)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.elements = _replace_var_list(self.elements, mapping)
        self.keys = [_replace_var(k, mapping) for k in self.keys]

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_container_create(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        if self.kind == ContainerKind.DICT:
            pairs = ", ".join(
                f"{k}: {v}" for k, v in zip(self.keys, self.elements)
            )
            return f"{pfx}{self.dst} = {{{pairs}}}"
        delim = {"LIST": "[]", "SET": "{{}}", "TUPLE": "()", "FROZENSET": "frozenset()"}
        d = delim.get(self.kind.name, "[]")
        elems = ", ".join(str(e) for e in self.elements)
        return f"{pfx}{self.dst} = {d[0]}{elems}{d[-1]}"


# ---------------------------------------------------------------------------

@dataclass
class UnpackNode(IRNode):
    """Destructuring assignment: ``targets = src``."""
    targets: List[SSAVar] = field(default_factory=list)
    src: SSAVar = field(default_factory=lambda: SSAVar("_"))
    star_index: Optional[int] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset(self.targets)

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.src})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.src = mapping.get(self.src, self.src)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_unpack(self)

    def pretty_print(self, indent: int = 0) -> str:
        tgts = ", ".join(
            f"*{t}" if i == self.star_index else str(t)
            for i, t in enumerate(self.targets)
        )
        return f"{self._indent(indent)}{tgts} = unpack({self.src})"


# ---------------------------------------------------------------------------

@dataclass
class YieldNode(IRNode):
    """Generator yield: ``dst = yield value``."""
    dst: Optional[SSAVar] = None
    value: Optional[SSAVar] = None
    is_yield_from: bool = False

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return _var_set(self.dst)

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return _var_set(self.value)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.value = _replace_var(self.value, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_yield(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        kw = "yield from" if self.is_yield_from else "yield"
        val = f" {self.value}" if self.value else ""
        dst = f"{self.dst} = " if self.dst else ""
        return f"{pfx}{dst}{kw}{val}"


# ---------------------------------------------------------------------------

@dataclass
class AwaitNode(IRNode):
    """Async await: ``dst = await value``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    value: SSAVar = field(default_factory=lambda: SSAVar("_"))

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.value})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.value = mapping.get(self.value, self.value)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_await(self)

    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.dst} = await {self.value}"


# ---------------------------------------------------------------------------

@dataclass
class RaiseNode(IRNode):
    """Exception raise: ``raise exc from cause``."""
    exc: Optional[SSAVar] = None
    cause: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = set()
        if self.exc is not None:
            vs.add(self.exc)
        if self.cause is not None:
            vs.add(self.cause)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.exc = _replace_var(self.exc, mapping)
        self.cause = _replace_var(self.cause, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_raise(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        if self.exc is None:
            return f"{pfx}raise"
        cause = f" from {self.cause}" if self.cause else ""
        return f"{pfx}raise {self.exc}{cause}"


# ---------------------------------------------------------------------------

@dataclass
class ExceptHandlerNode(IRNode):
    """Exception handler entry.

    ``exc_var = catch(exc_types…)``  — placed at the beginning of an
    except block.
    """
    exc_var: Optional[SSAVar] = None
    exc_types: List[str] = field(default_factory=list)
    handler_block: str = ""

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return _var_set(self.exc_var)

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        pass

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_except_handler(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        types = ", ".join(self.exc_types) or "*"
        dst = f"{self.exc_var} = " if self.exc_var else ""
        return f"{pfx}{dst}catch({types}) -> {self.handler_block}"


# ---------------------------------------------------------------------------

@dataclass
class ImportNode(IRNode):
    """Module import: ``dst = import(module_path)``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    module_path: str = ""
    names: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    is_from: bool = False
    level: int = 0

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        pass

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_import(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        if self.is_from:
            dots = "." * self.level
            names = ", ".join(
                f"{n} as {a}" if a else n for n, a in self.names
            )
            return f"{pfx}{self.dst} = from {dots}{self.module_path} import {names}"
        return f"{pfx}{self.dst} = import {self.module_path}"


# ---------------------------------------------------------------------------

@dataclass
class DeleteNode(IRNode):
    """Variable or attribute deletion: ``del target``."""
    target: SSAVar = field(default_factory=lambda: SSAVar("_"))
    attr: Optional[str] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.target})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.target = mapping.get(self.target, self.target)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_delete(self)

    def pretty_print(self, indent: int = 0) -> str:
        attr = f".{self.attr}" if self.attr else ""
        return f"{self._indent(indent)}del {self.target}{attr}"


# ---------------------------------------------------------------------------

@dataclass
class AssertNode(IRNode):
    """Assertion: ``assert test, msg``."""
    test: SSAVar = field(default_factory=lambda: SSAVar("_"))
    msg: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset()

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = {self.test}
        if self.msg is not None:
            vs.add(self.msg)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.test = mapping.get(self.test, self.test)
        self.msg = _replace_var(self.msg, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_assert(self)

    def pretty_print(self, indent: int = 0) -> str:
        m = f", {self.msg}" if self.msg else ""
        return f"{self._indent(indent)}assert {self.test}{m}"


# ---------------------------------------------------------------------------

@dataclass
class SliceNode(IRNode):
    """Slice operation: ``dst = obj[lower:upper:step]``."""
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    obj: SSAVar = field(default_factory=lambda: SSAVar("_"))
    lower: Optional[SSAVar] = None
    upper: Optional[SSAVar] = None
    step: Optional[SSAVar] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        vs: Set[SSAVar] = {self.obj}
        for v in (self.lower, self.upper, self.step):
            if v is not None:
                vs.add(v)
        return frozenset(vs)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.obj = mapping.get(self.obj, self.obj)
        self.lower = _replace_var(self.lower, mapping)
        self.upper = _replace_var(self.upper, mapping)
        self.step = _replace_var(self.step, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_slice(self)

    def pretty_print(self, indent: int = 0) -> str:
        lo = str(self.lower) if self.lower else ""
        hi = str(self.upper) if self.upper else ""
        st = str(self.step) if self.step else ""
        sl = f"{lo}:{hi}" + (f":{st}" if st else "")
        return f"{self._indent(indent)}{self.dst} = {self.obj}[{sl}]"


# ---------------------------------------------------------------------------

@dataclass
class FormatStringNode(IRNode):
    """F-string / template literal: ``dst = f"…{parts}…"``

    *parts* is a list of SSA vars whose string representations are
    concatenated.  *format_specs* optionally maps indices to format
    specifiers (e.g. ``:.2f``).  *static_parts* holds the literal
    string segments between interpolation holes.
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    parts: List[SSAVar] = field(default_factory=list)
    static_parts: List[str] = field(default_factory=list)
    format_specs: Dict[int, str] = field(default_factory=dict)

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset(self.parts)

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.parts = _replace_var_list(self.parts, mapping)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_format_string(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        segments: List[str] = []
        for i, part in enumerate(self.parts):
            if i < len(self.static_parts):
                segments.append(self.static_parts[i])
            spec = self.format_specs.get(i, "")
            segments.append(f"{{{part}{':' + spec if spec else ''}}}")
        if len(self.static_parts) > len(self.parts):
            segments.append(self.static_parts[-1])
        return f"{pfx}{self.dst} = f\"{''.join(segments)}\""


# ---------------------------------------------------------------------------

@dataclass
class ClosureCaptureNode(IRNode):
    """Closure variable capture.

    ``dst = capture(outer_var)``  — records that a variable from an
    enclosing scope is captured in the current closure.
    """
    dst: SSAVar = field(default_factory=lambda: SSAVar("_"))
    outer_var: SSAVar = field(default_factory=lambda: SSAVar("_"))
    is_nonlocal: bool = False
    enclosing_scope: Optional[str] = None

    @property
    def defined_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.dst})

    @property
    def used_vars(self) -> FrozenSet[SSAVar]:
        return frozenset({self.outer_var})

    def replace_uses(self, mapping: Mapping[SSAVar, SSAVar]) -> None:
        self.outer_var = mapping.get(self.outer_var, self.outer_var)

    def accept(self, visitor: IRVisitor) -> Any:
        return visitor.visit_closure_capture(self)

    def pretty_print(self, indent: int = 0) -> str:
        pfx = self._indent(indent)
        kw = "nonlocal " if self.is_nonlocal else ""
        scope = f" from {self.enclosing_scope}" if self.enclosing_scope else ""
        return f"{pfx}{self.dst} = capture({kw}{self.outer_var}{scope})"


# ---------------------------------------------------------------------------
# Utility: walk / collect helpers
# ---------------------------------------------------------------------------

def walk_nodes(nodes: Sequence[IRNode]) -> List[IRNode]:
    """Return a flat list of all nodes (identity — provided for API
    symmetry with tree-structured IRs)."""
    return list(nodes)


def collect_defined(nodes: Sequence[IRNode]) -> FrozenSet[SSAVar]:
    """Collect every SSA variable defined across *nodes*."""
    result: Set[SSAVar] = set()
    for n in nodes:
        result |= n.defined_vars
    return frozenset(result)


def collect_used(nodes: Sequence[IRNode]) -> FrozenSet[SSAVar]:
    """Collect every SSA variable used across *nodes*."""
    result: Set[SSAVar] = set()
    for n in nodes:
        result |= n.used_vars
    return frozenset(result)


def replace_all_uses(
    nodes: Sequence[IRNode],
    mapping: Mapping[SSAVar, SSAVar],
) -> None:
    """Rewrite uses across all *nodes* according to *mapping*."""
    for n in nodes:
        n.replace_uses(mapping)


def pretty_print_nodes(nodes: Sequence[IRNode], indent: int = 0) -> str:
    """Pretty-print a sequence of IR nodes."""
    return "\n".join(n.pretty_print(indent) for n in nodes)


# ---------------------------------------------------------------------------
# Node type registry (useful for serialisation / deserialisation)
# ---------------------------------------------------------------------------

_NODE_TYPE_REGISTRY: Dict[str, type] = {}


def _register_node_types() -> None:
    import inspect

    for name, obj in globals().items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, IRNode)
            and obj is not IRNode
        ):
            _NODE_TYPE_REGISTRY[name] = obj


_register_node_types()


def node_type_by_name(name: str) -> type:
    """Look up a concrete ``IRNode`` subclass by its class name."""
    return _NODE_TYPE_REGISTRY[name]


def all_node_types() -> Dict[str, type]:
    """Return a copy of the node-type registry."""
    return dict(_NODE_TYPE_REGISTRY)
