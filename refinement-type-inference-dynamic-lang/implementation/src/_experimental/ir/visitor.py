from __future__ import annotations

import copy
import enum
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Local lightweight IR types – no cross-module imports
# ---------------------------------------------------------------------------

@dataclass
class SourceRange:
    """Source location span."""
    file: str = ""
    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"


@dataclass
class SSAVar:
    """SSA variable with name and version."""
    name: str
    version: int = 0

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSAVar):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __str__(self) -> str:
        return f"{self.name}_{self.version}"

    def __repr__(self) -> str:
        return f"SSAVar({self.name!r}, {self.version})"


class TypeExpr:
    """Base type expression for the refinement type system."""

    def __init__(self, name: str = "Any") -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TypeExpr({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeExpr):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class RefinementPredicate:
    """Predicate for refinement types."""
    kind: str = ""  # isinstance, typeof, none_check, hasattr, comparison, truthiness
    subject: Optional[SSAVar] = None
    args: List[Any] = field(default_factory=list)
    negated: bool = False

    def negate(self) -> RefinementPredicate:
        return RefinementPredicate(
            kind=self.kind,
            subject=self.subject,
            args=list(self.args),
            negated=not self.negated,
        )

    def __str__(self) -> str:
        neg = "¬" if self.negated else ""
        args_s = ", ".join(str(a) for a in self.args)
        subj = str(self.subject) if self.subject else "?"
        return f"{neg}{self.kind}({subj}, {args_s})"


# ---------------------------------------------------------------------------
# IR Node hierarchy
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    """Base IR node."""
    source_range: Optional[SourceRange] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _node_id: int = field(default=0, repr=False)

    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        IRNode._counter += 1
        self._node_id = IRNode._counter

    def children(self) -> List[IRNode]:
        return []


@dataclass
class Expression(IRNode):
    """Base expression."""
    pass


@dataclass
class Statement(IRNode):
    """Base statement."""
    pass


@dataclass
class Constant(Expression):
    value: Any = None
    type_expr: Optional[TypeExpr] = None

    def __str__(self) -> str:
        return repr(self.value)


@dataclass
class Name(Expression):
    var: Optional[SSAVar] = None

    def __str__(self) -> str:
        return str(self.var) if self.var else "?"


@dataclass
class BinaryOp(Expression):
    op: str = ""
    left: Optional[Expression] = None
    right: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryOp(Expression):
    op: str = ""
    operand: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.operand] if self.operand else []

    def __str__(self) -> str:
        return f"({self.op} {self.operand})"


@dataclass
class Compare(Expression):
    op: str = ""
    left: Optional[Expression] = None
    right: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class Call(Expression):
    func: Optional[Expression] = None
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.func:
            result.append(self.func)
        result.extend(self.args)
        result.extend(self.kwargs.values())
        return result

    def __str__(self) -> str:
        args_s = ", ".join(str(a) for a in self.args)
        kw = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_s, kw]))
        return f"{self.func}({all_args})"


@dataclass
class Attribute(Expression):
    value: Optional[Expression] = None
    attr: str = ""

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []

    def __str__(self) -> str:
        return f"{self.value}.{self.attr}"


@dataclass
class Subscript(Expression):
    value: Optional[Expression] = None
    index: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.value:
            result.append(self.value)
        if self.index:
            result.append(self.index)
        return result

    def __str__(self) -> str:
        return f"{self.value}[{self.index}]"


@dataclass
class Index(Expression):
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []


@dataclass
class Slice(Expression):
    lower: Optional[Expression] = None
    upper: Optional[Expression] = None
    step: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.lower:
            result.append(self.lower)
        if self.upper:
            result.append(self.upper)
        if self.step:
            result.append(self.step)
        return result


@dataclass
class Starred(Expression):
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []


@dataclass
class FormattedValue(Expression):
    value: Optional[Expression] = None
    conversion: int = -1
    format_spec: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.value:
            result.append(self.value)
        if self.format_spec:
            result.append(self.format_spec)
        return result


@dataclass
class TupleExpr(Expression):
    elts: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        return list(self.elts)

    def __str__(self) -> str:
        return f"({', '.join(str(e) for e in self.elts)})"


@dataclass
class ListExpr(Expression):
    elts: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        return list(self.elts)

    def __str__(self) -> str:
        return f"[{', '.join(str(e) for e in self.elts)}]"


@dataclass
class DictExpr(Expression):
    keys: List[Optional[Expression]] = field(default_factory=list)
    values: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        for k in self.keys:
            if k:
                result.append(k)
        result.extend(self.values)
        return result


@dataclass
class SetExpr(Expression):
    elts: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        return list(self.elts)


@dataclass
class Lambda(Expression):
    args: List[str] = field(default_factory=list)
    body: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.body] if self.body else []


@dataclass
class Comprehension(Expression):
    elt: Optional[Expression] = None
    generators: List[Any] = field(default_factory=list)
    is_async: bool = False

    def children(self) -> List[IRNode]:
        return [self.elt] if self.elt else []


# --- Statements ---

@dataclass
class Assign(Statement):
    target: Optional[SSAVar] = None
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []

    def __str__(self) -> str:
        return f"{self.target} = {self.value}"


@dataclass
class PhiNode(Statement):
    dest: Optional[SSAVar] = None
    incoming: Dict[str, SSAVar] = field(default_factory=dict)

    def children(self) -> List[IRNode]:
        return []

    def __str__(self) -> str:
        pairs = ", ".join(f"[{b}]: {v}" for b, v in self.incoming.items())
        return f"{self.dest} = phi({pairs})"


@dataclass
class Guard(Statement):
    predicate: Optional[RefinementPredicate] = None
    target: Optional[SSAVar] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None

    def __str__(self) -> str:
        return f"guard({self.predicate}) -> {self.true_branch}, {self.false_branch}"


@dataclass
class Return(Statement):
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []

    def __str__(self) -> str:
        return f"return {self.value}" if self.value else "return"


@dataclass
class If(Statement):
    test: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)
    orelse: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.test:
            result.append(self.test)
        result.extend(self.body)
        result.extend(self.orelse)
        return result


@dataclass
class While(Statement):
    test: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)
    orelse: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.test:
            result.append(self.test)
        result.extend(self.body)
        result.extend(self.orelse)
        return result


@dataclass
class For(Statement):
    target: Optional[SSAVar] = None
    iter_expr: Optional[Expression] = None
    body: List[Statement] = field(default_factory=list)
    orelse: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.iter_expr:
            result.append(self.iter_expr)
        result.extend(self.body)
        result.extend(self.orelse)
        return result


@dataclass
class With(Statement):
    items: List[Expression] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = list(self.items)
        result.extend(self.body)
        return result


@dataclass
class Try(Statement):
    body: List[Statement] = field(default_factory=list)
    handlers: List[Any] = field(default_factory=list)
    orelse: List[Statement] = field(default_factory=list)
    finalbody: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = list(self.body)
        result.extend(self.orelse)
        result.extend(self.finalbody)
        return result


@dataclass
class Raise(Statement):
    exc: Optional[Expression] = None
    cause: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.exc:
            result.append(self.exc)
        if self.cause:
            result.append(self.cause)
        return result


@dataclass
class Yield(Expression):
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []


@dataclass
class Await(Expression):
    value: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        return [self.value] if self.value else []


@dataclass
class FunctionDef(Statement):
    name: str = ""
    params: List[str] = field(default_factory=list)
    param_annotations: Dict[str, Any] = field(default_factory=dict)
    return_annotation: Optional[Any] = None
    body: List[Statement] = field(default_factory=list)
    decorators: List[Expression] = field(default_factory=list)
    is_async: bool = False

    def children(self) -> List[IRNode]:
        result: List[IRNode] = list(self.decorators)
        result.extend(self.body)
        return result


@dataclass
class ClassDef(Statement):
    name: str = ""
    bases: List[Expression] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    decorators: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        result: List[IRNode] = list(self.bases)
        result.extend(self.decorators)
        result.extend(self.body)
        return result


@dataclass
class Module(IRNode):
    body: List[Statement] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        return list(self.body)


@dataclass
class Import(Statement):
    names: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    module: Optional[str] = None
    is_from: bool = False

    def __str__(self) -> str:
        if self.is_from:
            ns = ", ".join(n if a is None else f"{n} as {a}" for n, a in self.names)
            return f"from {self.module} import {ns}"
        ns = ", ".join(n if a is None else f"{n} as {a}" for n, a in self.names)
        return f"import {ns}"


@dataclass
class Assert(Statement):
    test: Optional[Expression] = None
    msg: Optional[Expression] = None

    def children(self) -> List[IRNode]:
        result: List[IRNode] = []
        if self.test:
            result.append(self.test)
        if self.msg:
            result.append(self.msg)
        return result


@dataclass
class Delete(Statement):
    targets: List[Expression] = field(default_factory=list)

    def children(self) -> List[IRNode]:
        return list(self.targets)


@dataclass
class Global(Statement):
    names: List[str] = field(default_factory=list)


@dataclass
class Nonlocal(Statement):
    names: List[str] = field(default_factory=list)


@dataclass
class Break(Statement):
    pass


@dataclass
class Continue(Statement):
    pass


@dataclass
class Pass(Statement):
    pass


# ===================================================================
# IRVisitor
# ===================================================================

class IRVisitor:
    """Base visitor with double dispatch for IR nodes."""

    def visit(self, node: IRNode) -> Any:
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: IRNode) -> Any:
        for child in node.children():
            self.visit(child)
        return None

    def visit_Assign(self, node: Assign) -> Any:
        return self.generic_visit(node)

    def visit_PhiNode(self, node: PhiNode) -> Any:
        return self.generic_visit(node)

    def visit_Guard(self, node: Guard) -> Any:
        return self.generic_visit(node)

    def visit_Call(self, node: Call) -> Any:
        return self.generic_visit(node)

    def visit_Return(self, node: Return) -> Any:
        return self.generic_visit(node)

    def visit_BinaryOp(self, node: BinaryOp) -> Any:
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        return self.generic_visit(node)

    def visit_Compare(self, node: Compare) -> Any:
        return self.generic_visit(node)

    def visit_Attribute(self, node: Attribute) -> Any:
        return self.generic_visit(node)

    def visit_Subscript(self, node: Subscript) -> Any:
        return self.generic_visit(node)

    def visit_Index(self, node: Index) -> Any:
        return self.generic_visit(node)

    def visit_If(self, node: If) -> Any:
        return self.generic_visit(node)

    def visit_While(self, node: While) -> Any:
        return self.generic_visit(node)

    def visit_For(self, node: For) -> Any:
        return self.generic_visit(node)

    def visit_With(self, node: With) -> Any:
        return self.generic_visit(node)

    def visit_Try(self, node: Try) -> Any:
        return self.generic_visit(node)

    def visit_Raise(self, node: Raise) -> Any:
        return self.generic_visit(node)

    def visit_Yield(self, node: Yield) -> Any:
        return self.generic_visit(node)

    def visit_Await(self, node: Await) -> Any:
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ClassDef) -> Any:
        return self.generic_visit(node)

    def visit_Module(self, node: Module) -> Any:
        return self.generic_visit(node)

    def visit_Import(self, node: Import) -> Any:
        return self.generic_visit(node)

    def visit_Lambda(self, node: Lambda) -> Any:
        return self.generic_visit(node)

    def visit_Comprehension(self, node: Comprehension) -> Any:
        return self.generic_visit(node)

    def visit_Constant(self, node: Constant) -> Any:
        return self.generic_visit(node)

    def visit_Name(self, node: Name) -> Any:
        return self.generic_visit(node)

    def visit_TupleExpr(self, node: TupleExpr) -> Any:
        return self.generic_visit(node)

    def visit_ListExpr(self, node: ListExpr) -> Any:
        return self.generic_visit(node)

    def visit_DictExpr(self, node: DictExpr) -> Any:
        return self.generic_visit(node)

    def visit_SetExpr(self, node: SetExpr) -> Any:
        return self.generic_visit(node)

    def visit_Slice(self, node: Slice) -> Any:
        return self.generic_visit(node)

    def visit_Starred(self, node: Starred) -> Any:
        return self.generic_visit(node)

    def visit_FormattedValue(self, node: FormattedValue) -> Any:
        return self.generic_visit(node)

    def visit_Assert(self, node: Assert) -> Any:
        return self.generic_visit(node)

    def visit_Delete(self, node: Delete) -> Any:
        return self.generic_visit(node)

    def visit_Global(self, node: Global) -> Any:
        return self.generic_visit(node)

    def visit_Nonlocal(self, node: Nonlocal) -> Any:
        return self.generic_visit(node)

    def visit_Break(self, node: Break) -> Any:
        return self.generic_visit(node)

    def visit_Continue(self, node: Continue) -> Any:
        return self.generic_visit(node)

    def visit_Pass(self, node: Pass) -> Any:
        return self.generic_visit(node)


# ===================================================================
# IRTransformer
# ===================================================================

class IRTransformer:
    """Base transformer – returns new/modified IR nodes."""

    def transform(self, node: IRNode) -> IRNode:
        method_name = f"transform_{type(node).__name__}"
        transformer = getattr(self, method_name, self.generic_transform)
        return transformer(node)

    def generic_transform(self, node: IRNode) -> IRNode:
        return node

    def _transform_list(self, nodes: List[IRNode]) -> List[IRNode]:
        return [self.transform(n) for n in nodes]

    def _transform_stmt_list(self, stmts: List[Statement]) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            t = self.transform(s)
            if isinstance(t, Statement):
                result.append(t)
        return result

    def _transform_expr_list(self, exprs: List[Expression]) -> List[Expression]:
        result: List[Expression] = []
        for e in exprs:
            t = self.transform(e)
            if isinstance(t, Expression):
                result.append(t)
        return result

    def transform_Assign(self, node: Assign) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_PhiNode(self, node: PhiNode) -> IRNode:
        return node

    def transform_Guard(self, node: Guard) -> IRNode:
        return node

    def transform_Call(self, node: Call) -> IRNode:
        new_func = self.transform(node.func) if node.func else None
        new_args = self._transform_expr_list(node.args)
        new_kwargs = {k: self.transform(v) for k, v in node.kwargs.items()}  # type: ignore[misc]
        if new_func is node.func and new_args == node.args:
            return node
        result = copy.copy(node)
        result.func = new_func  # type: ignore[assignment]
        result.args = new_args
        result.kwargs = new_kwargs  # type: ignore[assignment]
        return result

    def transform_Return(self, node: Return) -> IRNode:
        if node.value is None:
            return node
        new_val = self.transform(node.value)
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_BinaryOp(self, node: BinaryOp) -> IRNode:
        new_left = self.transform(node.left) if node.left else None
        new_right = self.transform(node.right) if node.right else None
        if new_left is node.left and new_right is node.right:
            return node
        result = copy.copy(node)
        result.left = new_left  # type: ignore[assignment]
        result.right = new_right  # type: ignore[assignment]
        return result

    def transform_UnaryOp(self, node: UnaryOp) -> IRNode:
        new_op = self.transform(node.operand) if node.operand else None
        if new_op is node.operand:
            return node
        result = copy.copy(node)
        result.operand = new_op  # type: ignore[assignment]
        return result

    def transform_Compare(self, node: Compare) -> IRNode:
        new_left = self.transform(node.left) if node.left else None
        new_right = self.transform(node.right) if node.right else None
        if new_left is node.left and new_right is node.right:
            return node
        result = copy.copy(node)
        result.left = new_left  # type: ignore[assignment]
        result.right = new_right  # type: ignore[assignment]
        return result

    def transform_Attribute(self, node: Attribute) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_Subscript(self, node: Subscript) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        new_idx = self.transform(node.index) if node.index else None
        if new_val is node.value and new_idx is node.index:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        result.index = new_idx  # type: ignore[assignment]
        return result

    def transform_Index(self, node: Index) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_If(self, node: If) -> IRNode:
        new_test = self.transform(node.test) if node.test else None
        new_body = self._transform_stmt_list(node.body)
        new_else = self._transform_stmt_list(node.orelse)
        result = copy.copy(node)
        result.test = new_test  # type: ignore[assignment]
        result.body = new_body
        result.orelse = new_else
        return result

    def transform_While(self, node: While) -> IRNode:
        new_test = self.transform(node.test) if node.test else None
        new_body = self._transform_stmt_list(node.body)
        new_else = self._transform_stmt_list(node.orelse)
        result = copy.copy(node)
        result.test = new_test  # type: ignore[assignment]
        result.body = new_body
        result.orelse = new_else
        return result

    def transform_For(self, node: For) -> IRNode:
        new_iter = self.transform(node.iter_expr) if node.iter_expr else None
        new_body = self._transform_stmt_list(node.body)
        new_else = self._transform_stmt_list(node.orelse)
        result = copy.copy(node)
        result.iter_expr = new_iter  # type: ignore[assignment]
        result.body = new_body
        result.orelse = new_else
        return result

    def transform_With(self, node: With) -> IRNode:
        new_items = self._transform_expr_list(node.items)
        new_body = self._transform_stmt_list(node.body)
        result = copy.copy(node)
        result.items = new_items
        result.body = new_body
        return result

    def transform_Try(self, node: Try) -> IRNode:
        new_body = self._transform_stmt_list(node.body)
        new_else = self._transform_stmt_list(node.orelse)
        new_final = self._transform_stmt_list(node.finalbody)
        result = copy.copy(node)
        result.body = new_body
        result.orelse = new_else
        result.finalbody = new_final
        return result

    def transform_Raise(self, node: Raise) -> IRNode:
        new_exc = self.transform(node.exc) if node.exc else None
        new_cause = self.transform(node.cause) if node.cause else None
        if new_exc is node.exc and new_cause is node.cause:
            return node
        result = copy.copy(node)
        result.exc = new_exc  # type: ignore[assignment]
        result.cause = new_cause  # type: ignore[assignment]
        return result

    def transform_Yield(self, node: Yield) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_Await(self, node: Await) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_FunctionDef(self, node: FunctionDef) -> IRNode:
        new_body = self._transform_stmt_list(node.body)
        new_decs = self._transform_expr_list(node.decorators)
        result = copy.copy(node)
        result.body = new_body
        result.decorators = new_decs
        return result

    def transform_ClassDef(self, node: ClassDef) -> IRNode:
        new_bases = self._transform_expr_list(node.bases)
        new_body = self._transform_stmt_list(node.body)
        new_decs = self._transform_expr_list(node.decorators)
        result = copy.copy(node)
        result.bases = new_bases
        result.body = new_body
        result.decorators = new_decs
        return result

    def transform_Module(self, node: Module) -> IRNode:
        new_body = self._transform_stmt_list(node.body)
        result = copy.copy(node)
        result.body = new_body
        return result

    def transform_Import(self, node: Import) -> IRNode:
        return node

    def transform_Lambda(self, node: Lambda) -> IRNode:
        new_body = self.transform(node.body) if node.body else None
        if new_body is node.body:
            return node
        result = copy.copy(node)
        result.body = new_body  # type: ignore[assignment]
        return result

    def transform_Comprehension(self, node: Comprehension) -> IRNode:
        new_elt = self.transform(node.elt) if node.elt else None
        if new_elt is node.elt:
            return node
        result = copy.copy(node)
        result.elt = new_elt  # type: ignore[assignment]
        return result

    def transform_Constant(self, node: Constant) -> IRNode:
        return node

    def transform_Name(self, node: Name) -> IRNode:
        return node

    def transform_TupleExpr(self, node: TupleExpr) -> IRNode:
        new_elts = self._transform_expr_list(node.elts)
        result = copy.copy(node)
        result.elts = new_elts
        return result

    def transform_ListExpr(self, node: ListExpr) -> IRNode:
        new_elts = self._transform_expr_list(node.elts)
        result = copy.copy(node)
        result.elts = new_elts
        return result

    def transform_DictExpr(self, node: DictExpr) -> IRNode:
        new_keys: List[Optional[Expression]] = []
        for k in node.keys:
            if k is not None:
                t = self.transform(k)
                new_keys.append(t if isinstance(t, Expression) else k)
            else:
                new_keys.append(None)
        new_vals = self._transform_expr_list(node.values)
        result = copy.copy(node)
        result.keys = new_keys
        result.values = new_vals
        return result

    def transform_SetExpr(self, node: SetExpr) -> IRNode:
        new_elts = self._transform_expr_list(node.elts)
        result = copy.copy(node)
        result.elts = new_elts
        return result

    def transform_Slice(self, node: Slice) -> IRNode:
        new_lower = self.transform(node.lower) if node.lower else None
        new_upper = self.transform(node.upper) if node.upper else None
        new_step = self.transform(node.step) if node.step else None
        result = copy.copy(node)
        result.lower = new_lower  # type: ignore[assignment]
        result.upper = new_upper  # type: ignore[assignment]
        result.step = new_step  # type: ignore[assignment]
        return result

    def transform_Starred(self, node: Starred) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        if new_val is node.value:
            return node
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        return result

    def transform_FormattedValue(self, node: FormattedValue) -> IRNode:
        new_val = self.transform(node.value) if node.value else None
        new_fmt = self.transform(node.format_spec) if node.format_spec else None
        result = copy.copy(node)
        result.value = new_val  # type: ignore[assignment]
        result.format_spec = new_fmt  # type: ignore[assignment]
        return result

    def transform_Assert(self, node: Assert) -> IRNode:
        new_test = self.transform(node.test) if node.test else None
        new_msg = self.transform(node.msg) if node.msg else None
        result = copy.copy(node)
        result.test = new_test  # type: ignore[assignment]
        result.msg = new_msg  # type: ignore[assignment]
        return result

    def transform_Delete(self, node: Delete) -> IRNode:
        new_targets = self._transform_expr_list(node.targets)
        result = copy.copy(node)
        result.targets = new_targets
        return result

    def transform_Global(self, node: Global) -> IRNode:
        return node

    def transform_Nonlocal(self, node: Nonlocal) -> IRNode:
        return node

    def transform_Break(self, node: Break) -> IRNode:
        return node

    def transform_Continue(self, node: Continue) -> IRNode:
        return node

    def transform_Pass(self, node: Pass) -> IRNode:
        return node


# ===================================================================
# IRRewriter
# ===================================================================

class IRRewriter:
    """In-place rewriter for SSA variables in IR trees."""

    def rewrite_uses(self, node: IRNode, old_var: SSAVar, new_var: SSAVar) -> None:
        """Replace all uses of old_var with new_var in-place."""
        if isinstance(node, Name):
            if node.var == old_var:
                node.var = new_var
        elif isinstance(node, Assign):
            if node.value:
                self.rewrite_uses(node.value, old_var, new_var)
        elif isinstance(node, PhiNode):
            for bid in list(node.incoming.keys()):
                if node.incoming[bid] == old_var:
                    node.incoming[bid] = new_var
        elif isinstance(node, Guard):
            if node.predicate and node.predicate.subject == old_var:
                node.predicate.subject = new_var
        else:
            for child in node.children():
                self.rewrite_uses(child, old_var, new_var)

    def rewrite_defs(self, node: IRNode, old_var: SSAVar, new_var: SSAVar) -> None:
        """Replace all definitions of old_var with new_var in-place."""
        if isinstance(node, Assign):
            if node.target == old_var:
                node.target = new_var
        elif isinstance(node, PhiNode):
            if node.dest == old_var:
                node.dest = new_var
        elif isinstance(node, For):
            if node.target == old_var:
                node.target = new_var
        for child in node.children():
            self.rewrite_defs(child, old_var, new_var)

    def substitute(self, node: IRNode, mapping: Dict[SSAVar, Expression]) -> None:
        """Substitute SSA variables with expressions according to mapping."""
        if isinstance(node, Name):
            if node.var and node.var in mapping:
                replacement = mapping[node.var]
                if isinstance(replacement, Name):
                    node.var = replacement.var
        elif isinstance(node, Assign):
            if node.value:
                self.substitute(node.value, mapping)
        elif isinstance(node, BinaryOp):
            if node.left:
                self._substitute_expr_field(node, "left", mapping)
            if node.right:
                self._substitute_expr_field(node, "right", mapping)
        elif isinstance(node, UnaryOp):
            if node.operand:
                self._substitute_expr_field(node, "operand", mapping)
        elif isinstance(node, Compare):
            if node.left:
                self._substitute_expr_field(node, "left", mapping)
            if node.right:
                self._substitute_expr_field(node, "right", mapping)
        elif isinstance(node, Call):
            if node.func:
                self._substitute_expr_field(node, "func", mapping)
            for i, arg in enumerate(node.args):
                if isinstance(arg, Name) and arg.var and arg.var in mapping:
                    node.args[i] = mapping[arg.var]
                else:
                    self.substitute(arg, mapping)
            for k, v in list(node.kwargs.items()):
                if isinstance(v, Name) and v.var and v.var in mapping:
                    node.kwargs[k] = mapping[v.var]
                else:
                    self.substitute(v, mapping)
        elif isinstance(node, Return):
            if node.value:
                self._substitute_expr_field(node, "value", mapping)
        elif isinstance(node, Attribute):
            if node.value:
                self._substitute_expr_field(node, "value", mapping)
        elif isinstance(node, Subscript):
            if node.value:
                self._substitute_expr_field(node, "value", mapping)
            if node.index:
                self._substitute_expr_field(node, "index", mapping)
        else:
            for child in node.children():
                self.substitute(child, mapping)

    def _substitute_expr_field(
        self, node: IRNode, field_name: str, mapping: Dict[SSAVar, Expression]
    ) -> None:
        val = getattr(node, field_name, None)
        if isinstance(val, Name) and val.var and val.var in mapping:
            setattr(node, field_name, mapping[val.var])
        elif val is not None:
            self.substitute(val, mapping)


# ===================================================================
# IRCollector
# ===================================================================

class IRCollector:
    """Collects IR nodes matching predicates."""

    def collect_all(self, root: IRNode, predicate: Callable[[IRNode], bool]) -> List[IRNode]:
        """Collect all nodes matching the predicate."""
        results: List[IRNode] = []
        self._walk(root, predicate, results)
        return results

    def _walk(self, node: IRNode, predicate: Callable[[IRNode], bool], acc: List[IRNode]) -> None:
        if predicate(node):
            acc.append(node)
        for child in node.children():
            self._walk(child, predicate, acc)

    def collect_defs(self, root: IRNode, var: SSAVar) -> List[IRNode]:
        """Collect all definitions of a variable."""
        def pred(n: IRNode) -> bool:
            if isinstance(n, Assign) and n.target == var:
                return True
            if isinstance(n, PhiNode) and n.dest == var:
                return True
            if isinstance(n, For) and n.target == var:
                return True
            return False
        return self.collect_all(root, pred)

    def collect_uses(self, root: IRNode, var: SSAVar) -> List[IRNode]:
        """Collect all uses of a variable."""
        def pred(n: IRNode) -> bool:
            if isinstance(n, Name) and n.var == var:
                return True
            if isinstance(n, PhiNode):
                return var in n.incoming.values()
            return False
        return self.collect_all(root, pred)

    def collect_guards(self, root: IRNode) -> List[Guard]:
        """Collect all guard nodes."""
        results: List[Guard] = []
        for n in self.collect_all(root, lambda x: isinstance(x, Guard)):
            assert isinstance(n, Guard)
            results.append(n)
        return results

    def collect_calls(self, root: IRNode) -> List[Call]:
        """Collect all call nodes."""
        results: List[Call] = []
        for n in self.collect_all(root, lambda x: isinstance(x, Call)):
            assert isinstance(n, Call)
            results.append(n)
        return results

    def collect_by_type(self, root: IRNode, node_type: Type[IRNode]) -> List[IRNode]:
        """Collect all nodes of a specific type."""
        return self.collect_all(root, lambda x: isinstance(x, node_type))

    def collect_names(self, root: IRNode) -> List[Name]:
        """Collect all Name nodes."""
        results: List[Name] = []
        for n in self.collect_all(root, lambda x: isinstance(x, Name)):
            assert isinstance(n, Name)
            results.append(n)
        return results

    def collect_constants(self, root: IRNode) -> List[Constant]:
        """Collect all Constant nodes."""
        results: List[Constant] = []
        for n in self.collect_all(root, lambda x: isinstance(x, Constant)):
            assert isinstance(n, Constant)
            results.append(n)
        return results

    def collect_assignments(self, root: IRNode) -> List[Assign]:
        """Collect all Assign nodes."""
        results: List[Assign] = []
        for n in self.collect_all(root, lambda x: isinstance(x, Assign)):
            assert isinstance(n, Assign)
            results.append(n)
        return results


# ===================================================================
# UseDefAnalyzer
# ===================================================================

class UseDefAnalyzer:
    """SSA use-def analysis over IR trees."""

    def __init__(self) -> None:
        self._defs: Dict[str, IRNode] = {}
        self._uses: Dict[str, List[IRNode]] = {}
        self._built = False

    def build(self, root: IRNode) -> None:
        """Build use-def and def-use chains from the IR tree."""
        self._defs = {}
        self._uses = {}
        self._build_walk(root)
        self._built = True

    def _build_walk(self, node: IRNode) -> None:
        # record definitions
        if isinstance(node, Assign) and node.target:
            key = str(node.target)
            self._defs[key] = node
        elif isinstance(node, PhiNode) and node.dest:
            key = str(node.dest)
            self._defs[key] = node
        elif isinstance(node, For) and node.target:
            key = str(node.target)
            self._defs[key] = node

        # record uses
        if isinstance(node, Name) and node.var:
            key = str(node.var)
            self._uses.setdefault(key, []).append(node)
        elif isinstance(node, PhiNode):
            for v in node.incoming.values():
                key = str(v)
                self._uses.setdefault(key, []).append(node)
        elif isinstance(node, Guard) and node.predicate and node.predicate.subject:
            key = str(node.predicate.subject)
            self._uses.setdefault(key, []).append(node)

        for child in node.children():
            self._build_walk(child)

    def get_def(self, var: SSAVar) -> Optional[IRNode]:
        """Get the single definition of an SSA variable."""
        return self._defs.get(str(var))

    def get_uses(self, var: SSAVar) -> List[IRNode]:
        """Get all uses of an SSA variable."""
        return self._uses.get(str(var), [])

    def get_def_use_chains(self) -> Dict[str, List[IRNode]]:
        """Get def-use chains: variable -> list of use nodes."""
        return dict(self._uses)

    def get_use_def_chains(self) -> Dict[str, IRNode]:
        """Get use-def chains: variable -> definition node."""
        return dict(self._defs)

    def is_used(self, var: SSAVar) -> bool:
        return len(self._uses.get(str(var), [])) > 0

    def use_count(self, var: SSAVar) -> int:
        return len(self._uses.get(str(var), []))

    def get_dead_defs(self) -> List[IRNode]:
        """Get definitions whose variables are never used."""
        dead: List[IRNode] = []
        for key, def_node in self._defs.items():
            if not self._uses.get(key):
                dead.append(def_node)
        return dead

    def get_undefined_uses(self) -> List[IRNode]:
        """Get uses of variables that have no definition."""
        undef: List[IRNode] = []
        for key, use_nodes in self._uses.items():
            if key not in self._defs:
                undef.extend(use_nodes)
        return undef


# ===================================================================
# IRPrinter
# ===================================================================

class IRPrinter:
    """Pretty-print IR with optional ANSI colors and type annotations."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    def __init__(
        self,
        show_types: bool = False,
        show_locations: bool = False,
        show_ssa_versions: bool = True,
        use_color: bool = True,
        indent_size: int = 2,
    ) -> None:
        self.show_types = show_types
        self.show_locations = show_locations
        self.show_ssa_versions = show_ssa_versions
        self.use_color = use_color
        self.indent_size = indent_size

    def _c(self, code: str, text: str) -> str:
        if self.use_color:
            return f"{code}{text}{self.RESET}"
        return text

    def _indent(self, depth: int) -> str:
        return " " * (self.indent_size * depth)

    def print_node(self, node: IRNode, depth: int = 0) -> str:
        method_name = f"_print_{type(node).__name__}"
        printer = getattr(self, method_name, self._print_generic)
        return printer(node, depth)

    def _print_generic(self, node: IRNode, depth: int) -> str:
        lines = [f"{self._indent(depth)}{self._c(self.GRAY, type(node).__name__)}"]
        for child in node.children():
            lines.append(self.print_node(child, depth + 1))
        return "\n".join(lines)

    def _loc_suffix(self, node: IRNode) -> str:
        if self.show_locations and node.source_range:
            return self._c(self.GRAY, f"  # {node.source_range}")
        return ""

    def _var_str(self, var: Optional[SSAVar]) -> str:
        if var is None:
            return "?"
        if self.show_ssa_versions:
            return self._c(self.CYAN, f"{var.name}_{var.version}")
        return self._c(self.CYAN, var.name)

    def _print_Module(self, node: Module, depth: int) -> str:
        lines = [f"{self._indent(depth)}{self._c(self.BOLD, 'module')}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_FunctionDef(self, node: FunctionDef, depth: int) -> str:
        ind = self._indent(depth)
        async_str = "async " if node.is_async else ""
        params_s = ", ".join(node.params)
        header = f"{ind}{self._c(self.BOLD, f'{async_str}def')} {self._c(self.YELLOW, node.name)}({params_s})"
        if node.return_annotation:
            header += f" -> {node.return_annotation}"
        header += ":"
        lines = [header]
        for dec in node.decorators:
            lines.append(f"{self._indent(depth + 1)}@{self.print_node(dec, 0)}")
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_ClassDef(self, node: ClassDef, depth: int) -> str:
        ind = self._indent(depth)
        bases_s = ", ".join(self.print_node(b, 0) for b in node.bases)
        header = f"{ind}{self._c(self.BOLD, 'class')} {self._c(self.YELLOW, node.name)}"
        if bases_s:
            header += f"({bases_s})"
        header += ":"
        lines = [header]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_Assign(self, node: Assign, depth: int) -> str:
        ind = self._indent(depth)
        target = self._var_str(node.target)
        value = self.print_node(node.value, 0) if node.value else "?"
        return f"{ind}{target} = {value}{self._loc_suffix(node)}"

    def _print_PhiNode(self, node: PhiNode, depth: int) -> str:
        ind = self._indent(depth)
        dest = self._var_str(node.dest)
        pairs = ", ".join(
            f"[{self._c(self.BLUE, bid)}]: {self._var_str(v)}"
            for bid, v in node.incoming.items()
        )
        return f"{ind}{dest} = {self._c(self.MAGENTA, 'phi')}({pairs}){self._loc_suffix(node)}"

    def _print_Guard(self, node: Guard, depth: int) -> str:
        ind = self._indent(depth)
        pred_s = str(node.predicate) if node.predicate else "?"
        return f"{ind}{self._c(self.RED, 'guard')}({pred_s}) -> {node.true_branch}, {node.false_branch}{self._loc_suffix(node)}"

    def _print_Return(self, node: Return, depth: int) -> str:
        ind = self._indent(depth)
        if node.value:
            val = self.print_node(node.value, 0)
            return f"{ind}{self._c(self.RED, 'return')} {val}{self._loc_suffix(node)}"
        return f"{ind}{self._c(self.RED, 'return')}{self._loc_suffix(node)}"

    def _print_If(self, node: If, depth: int) -> str:
        ind = self._indent(depth)
        test = self.print_node(node.test, 0) if node.test else "?"
        lines = [f"{ind}{self._c(self.BOLD, 'if')} {test}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        if node.orelse:
            lines.append(f"{ind}{self._c(self.BOLD, 'else')}:")
            for s in node.orelse:
                lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_While(self, node: While, depth: int) -> str:
        ind = self._indent(depth)
        test = self.print_node(node.test, 0) if node.test else "?"
        lines = [f"{ind}{self._c(self.BOLD, 'while')} {test}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_For(self, node: For, depth: int) -> str:
        ind = self._indent(depth)
        target = self._var_str(node.target)
        iter_s = self.print_node(node.iter_expr, 0) if node.iter_expr else "?"
        lines = [f"{ind}{self._c(self.BOLD, 'for')} {target} in {iter_s}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_With(self, node: With, depth: int) -> str:
        ind = self._indent(depth)
        items_s = ", ".join(self.print_node(it, 0) for it in node.items)
        lines = [f"{ind}{self._c(self.BOLD, 'with')} {items_s}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_Try(self, node: Try, depth: int) -> str:
        ind = self._indent(depth)
        lines = [f"{ind}{self._c(self.BOLD, 'try')}:"]
        for s in node.body:
            lines.append(self.print_node(s, depth + 1))
        if node.orelse:
            lines.append(f"{ind}{self._c(self.BOLD, 'else')}:")
            for s in node.orelse:
                lines.append(self.print_node(s, depth + 1))
        if node.finalbody:
            lines.append(f"{ind}{self._c(self.BOLD, 'finally')}:")
            for s in node.finalbody:
                lines.append(self.print_node(s, depth + 1))
        return "\n".join(lines)

    def _print_Raise(self, node: Raise, depth: int) -> str:
        ind = self._indent(depth)
        if node.exc:
            exc_s = self.print_node(node.exc, 0)
            if node.cause:
                cause_s = self.print_node(node.cause, 0)
                return f"{ind}{self._c(self.RED, 'raise')} {exc_s} from {cause_s}"
            return f"{ind}{self._c(self.RED, 'raise')} {exc_s}"
        return f"{ind}{self._c(self.RED, 'raise')}"

    def _print_BinaryOp(self, node: BinaryOp, depth: int) -> str:
        left = self.print_node(node.left, 0) if node.left else "?"
        right = self.print_node(node.right, 0) if node.right else "?"
        return f"({left} {self._c(self.YELLOW, node.op)} {right})"

    def _print_UnaryOp(self, node: UnaryOp, depth: int) -> str:
        operand = self.print_node(node.operand, 0) if node.operand else "?"
        return f"({self._c(self.YELLOW, node.op)}{operand})"

    def _print_Compare(self, node: Compare, depth: int) -> str:
        left = self.print_node(node.left, 0) if node.left else "?"
        right = self.print_node(node.right, 0) if node.right else "?"
        return f"({left} {self._c(self.YELLOW, node.op)} {right})"

    def _print_Call(self, node: Call, depth: int) -> str:
        func = self.print_node(node.func, 0) if node.func else "?"
        args_s = ", ".join(self.print_node(a, 0) for a in node.args)
        kw_s = ", ".join(f"{k}={self.print_node(v, 0)}" for k, v in node.kwargs.items())
        all_args = ", ".join(filter(None, [args_s, kw_s]))
        return f"{func}({all_args})"

    def _print_Attribute(self, node: Attribute, depth: int) -> str:
        val = self.print_node(node.value, 0) if node.value else "?"
        return f"{val}.{node.attr}"

    def _print_Subscript(self, node: Subscript, depth: int) -> str:
        val = self.print_node(node.value, 0) if node.value else "?"
        idx = self.print_node(node.index, 0) if node.index else "?"
        return f"{val}[{idx}]"

    def _print_Constant(self, node: Constant, depth: int) -> str:
        return self._c(self.GREEN, repr(node.value))

    def _print_Name(self, node: Name, depth: int) -> str:
        return self._var_str(node.var)

    def _print_TupleExpr(self, node: TupleExpr, depth: int) -> str:
        elts = ", ".join(self.print_node(e, 0) for e in node.elts)
        return f"({elts})"

    def _print_ListExpr(self, node: ListExpr, depth: int) -> str:
        elts = ", ".join(self.print_node(e, 0) for e in node.elts)
        return f"[{elts}]"

    def _print_DictExpr(self, node: DictExpr, depth: int) -> str:
        pairs: List[str] = []
        for k, v in zip(node.keys, node.values):
            k_s = self.print_node(k, 0) if k else "**"
            v_s = self.print_node(v, 0)
            pairs.append(f"{k_s}: {v_s}")
        return "{" + ", ".join(pairs) + "}"

    def _print_SetExpr(self, node: SetExpr, depth: int) -> str:
        elts = ", ".join(self.print_node(e, 0) for e in node.elts)
        return "{" + elts + "}"

    def _print_Import(self, node: Import, depth: int) -> str:
        ind = self._indent(depth)
        return f"{ind}{node}"

    def _print_Yield(self, node: Yield, depth: int) -> str:
        if node.value:
            val = self.print_node(node.value, 0)
            return f"{self._c(self.MAGENTA, 'yield')} {val}"
        return self._c(self.MAGENTA, "yield")

    def _print_Await(self, node: Await, depth: int) -> str:
        val = self.print_node(node.value, 0) if node.value else "?"
        return f"{self._c(self.MAGENTA, 'await')} {val}"

    def _print_Lambda(self, node: Lambda, depth: int) -> str:
        args_s = ", ".join(node.args)
        body = self.print_node(node.body, 0) if node.body else "?"
        return f"lambda {args_s}: {body}"

    def _print_Assert(self, node: Assert, depth: int) -> str:
        ind = self._indent(depth)
        test = self.print_node(node.test, 0) if node.test else "?"
        if node.msg:
            msg = self.print_node(node.msg, 0)
            return f"{ind}assert {test}, {msg}"
        return f"{ind}assert {test}"

    def _print_Delete(self, node: Delete, depth: int) -> str:
        ind = self._indent(depth)
        targets = ", ".join(self.print_node(t, 0) for t in node.targets)
        return f"{ind}del {targets}"

    def _print_Global(self, node: Global, depth: int) -> str:
        ind = self._indent(depth)
        return f"{ind}global {', '.join(node.names)}"

    def _print_Nonlocal(self, node: Nonlocal, depth: int) -> str:
        ind = self._indent(depth)
        return f"{ind}nonlocal {', '.join(node.names)}"

    def _print_Break(self, node: Break, depth: int) -> str:
        return f"{self._indent(depth)}{self._c(self.RED, 'break')}"

    def _print_Continue(self, node: Continue, depth: int) -> str:
        return f"{self._indent(depth)}{self._c(self.YELLOW, 'continue')}"

    def _print_Pass(self, node: Pass, depth: int) -> str:
        return f"{self._indent(depth)}{self._c(self.GRAY, 'pass')}"

    def print_function(self, node: FunctionDef) -> str:
        return self.print_node(node, 0)

    def print_type(self, type_expr: TypeExpr) -> str:
        return self._c(self.GREEN, str(type_expr))

    def print_predicate(self, pred: RefinementPredicate) -> str:
        return self._c(self.MAGENTA, str(pred))

    def print_refinement(self, var: SSAVar, pred: RefinementPredicate) -> str:
        return f"{self._var_str(var)} : {self.print_predicate(pred)}"


# ===================================================================
# IRValidator
# ===================================================================

@dataclass
class ValidationError:
    """A validation error."""
    message: str
    node: Optional[IRNode] = None
    severity: str = "error"  # error, warning

    def __str__(self) -> str:
        loc = ""
        if self.node and self.node.source_range:
            loc = f" at {self.node.source_range}"
        return f"[{self.severity}] {self.message}{loc}"


class IRValidator:
    """Validate IR well-formedness."""

    def __init__(self) -> None:
        self._errors: List[ValidationError] = []

    def validate(self, root: IRNode) -> List[ValidationError]:
        """Run all validation checks."""
        self._errors = []
        self.check_ssa_property(root)
        self.check_type_consistency(root)
        self.check_phi_consistency(root)
        self.check_guard_consistency(root)
        self.check_source_locations(root)
        return self._errors

    def _add_error(self, msg: str, node: Optional[IRNode] = None, severity: str = "error") -> None:
        self._errors.append(ValidationError(msg, node, severity))

    def check_ssa_property(self, root: IRNode) -> List[ValidationError]:
        """Check each SSA variable defined exactly once."""
        defs: Dict[str, List[IRNode]] = {}
        self._collect_defs_walk(root, defs)
        errors: List[ValidationError] = []
        for var_key, def_nodes in defs.items():
            if len(def_nodes) > 1:
                err = ValidationError(
                    f"SSA variable {var_key} defined {len(def_nodes)} times",
                    def_nodes[0],
                )
                errors.append(err)
                self._errors.append(err)

        # check uses have defs
        uses: Dict[str, List[IRNode]] = {}
        self._collect_uses_walk(root, uses)
        for var_key, use_nodes in uses.items():
            if var_key not in defs:
                err = ValidationError(
                    f"Variable {var_key} used but never defined",
                    use_nodes[0],
                    "warning",
                )
                errors.append(err)
                self._errors.append(err)

        return errors

    def _collect_defs_walk(self, node: IRNode, defs: Dict[str, List[IRNode]]) -> None:
        if isinstance(node, Assign) and node.target:
            key = str(node.target)
            defs.setdefault(key, []).append(node)
        elif isinstance(node, PhiNode) and node.dest:
            key = str(node.dest)
            defs.setdefault(key, []).append(node)
        elif isinstance(node, For) and node.target:
            key = str(node.target)
            defs.setdefault(key, []).append(node)
        for child in node.children():
            self._collect_defs_walk(child, defs)

    def _collect_uses_walk(self, node: IRNode, uses: Dict[str, List[IRNode]]) -> None:
        if isinstance(node, Name) and node.var:
            key = str(node.var)
            uses.setdefault(key, []).append(node)
        elif isinstance(node, PhiNode):
            for v in node.incoming.values():
                key = str(v)
                uses.setdefault(key, []).append(node)
        for child in node.children():
            self._collect_uses_walk(child, uses)

    def check_type_consistency(self, root: IRNode) -> List[ValidationError]:
        """Check type annotations are consistent with operations."""
        errors: List[ValidationError] = []
        self._check_types_walk(root, errors)
        return errors

    def _check_types_walk(self, node: IRNode, errors: List[ValidationError]) -> None:
        if isinstance(node, BinaryOp):
            if node.op in ("+", "-", "*", "/", "//", "%", "**"):
                # numeric ops: both operands should be compatible
                pass  # dynamic language, can't check statically without type info
        for child in node.children():
            self._check_types_walk(child, errors)

    def check_cfg_consistency(
        self, stmts: List[Statement], block_ids: Set[str]
    ) -> List[ValidationError]:
        """Check CFG edges match block structure."""
        errors: List[ValidationError] = []
        for stmt in stmts:
            if isinstance(stmt, Guard):
                if stmt.true_branch and stmt.true_branch not in block_ids:
                    err = ValidationError(
                        f"Guard true_branch {stmt.true_branch} not in CFG",
                        stmt,
                    )
                    errors.append(err)
                    self._errors.append(err)
                if stmt.false_branch and stmt.false_branch not in block_ids:
                    err = ValidationError(
                        f"Guard false_branch {stmt.false_branch} not in CFG",
                        stmt,
                    )
                    errors.append(err)
                    self._errors.append(err)
        return errors

    def check_phi_consistency(self, root: IRNode) -> List[ValidationError]:
        """Check phi node consistency."""
        errors: List[ValidationError] = []
        self._check_phi_walk(root, errors)
        return errors

    def _check_phi_walk(self, node: IRNode, errors: List[ValidationError]) -> None:
        if isinstance(node, PhiNode):
            if node.dest is None:
                err = ValidationError("Phi node has no destination", node)
                errors.append(err)
                self._errors.append(err)
            if not node.incoming:
                err = ValidationError(
                    f"Phi node {node.dest} has no incoming values",
                    node,
                    "warning",
                )
                errors.append(err)
                self._errors.append(err)
        for child in node.children():
            self._check_phi_walk(child, errors)

    def check_guard_consistency(self, root: IRNode) -> List[ValidationError]:
        """Check guard consistency."""
        errors: List[ValidationError] = []
        self._check_guard_walk(root, errors)
        return errors

    def _check_guard_walk(self, node: IRNode, errors: List[ValidationError]) -> None:
        if isinstance(node, Guard):
            if node.predicate is None:
                err = ValidationError("Guard has no predicate", node)
                errors.append(err)
                self._errors.append(err)
            if node.predicate and node.predicate.kind == "":
                err = ValidationError("Guard predicate has empty kind", node, "warning")
                errors.append(err)
                self._errors.append(err)
        for child in node.children():
            self._check_guard_walk(child, errors)

    def check_source_locations(self, root: IRNode) -> List[ValidationError]:
        """Check all nodes have source locations (warning-level)."""
        errors: List[ValidationError] = []
        self._check_locations_walk(root, errors)
        return errors

    def _check_locations_walk(self, node: IRNode, errors: List[ValidationError]) -> None:
        if node.source_range is None:
            if not isinstance(node, (Constant, Name, Pass, Break, Continue)):
                err = ValidationError(
                    f"{type(node).__name__} missing source location",
                    node,
                    "warning",
                )
                errors.append(err)
                # don't add to self._errors to avoid noise
        for child in node.children():
            self._check_locations_walk(child, errors)


# ===================================================================
# IRSerializer
# ===================================================================

class IRSerializer:
    """Serialize/deserialize IR to JSON/msgpack."""

    SCHEMA_VERSION = 1

    @staticmethod
    def to_json(node: IRNode) -> str:
        data = IRSerializer._to_dict(node)
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> IRNode:
        data = json.loads(json_str)
        return IRSerializer._from_dict(data)

    @staticmethod
    def to_msgpack(node: IRNode) -> bytes:
        """Serialize to msgpack (falls back to JSON bytes if msgpack not available)."""
        data = IRSerializer._to_dict(node)
        try:
            import msgpack
            return msgpack.packb(data, use_bin_type=True)
        except ImportError:
            return json.dumps(data).encode("utf-8")

    @staticmethod
    def from_msgpack(data: bytes) -> IRNode:
        """Deserialize from msgpack."""
        try:
            import msgpack
            d = msgpack.unpackb(data, raw=False)
        except ImportError:
            d = json.loads(data.decode("utf-8"))
        return IRSerializer._from_dict(d)

    @staticmethod
    def _to_dict(node: IRNode) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": type(node).__name__,
            "schema_version": IRSerializer.SCHEMA_VERSION,
        }

        if node.source_range:
            result["source_range"] = {
                "file": node.source_range.file,
                "start_line": node.source_range.start_line,
                "start_col": node.source_range.start_col,
                "end_line": node.source_range.end_line,
                "end_col": node.source_range.end_col,
            }

        if isinstance(node, Constant):
            result["value"] = node.value
            if node.type_expr:
                result["type_expr"] = str(node.type_expr)
        elif isinstance(node, Name):
            if node.var:
                result["var"] = {"name": node.var.name, "version": node.var.version}
        elif isinstance(node, BinaryOp):
            result["op"] = node.op
            if node.left:
                result["left"] = IRSerializer._to_dict(node.left)
            if node.right:
                result["right"] = IRSerializer._to_dict(node.right)
        elif isinstance(node, UnaryOp):
            result["op"] = node.op
            if node.operand:
                result["operand"] = IRSerializer._to_dict(node.operand)
        elif isinstance(node, Compare):
            result["op"] = node.op
            if node.left:
                result["left"] = IRSerializer._to_dict(node.left)
            if node.right:
                result["right"] = IRSerializer._to_dict(node.right)
        elif isinstance(node, Call):
            if node.func:
                result["func"] = IRSerializer._to_dict(node.func)
            result["args"] = [IRSerializer._to_dict(a) for a in node.args]
            result["kwargs"] = {k: IRSerializer._to_dict(v) for k, v in node.kwargs.items()}
        elif isinstance(node, Attribute):
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
            result["attr"] = node.attr
        elif isinstance(node, Subscript):
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
            if node.index:
                result["index"] = IRSerializer._to_dict(node.index)
        elif isinstance(node, Assign):
            if node.target:
                result["target"] = {"name": node.target.name, "version": node.target.version}
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
        elif isinstance(node, PhiNode):
            if node.dest:
                result["dest"] = {"name": node.dest.name, "version": node.dest.version}
            result["incoming"] = {
                k: {"name": v.name, "version": v.version}
                for k, v in node.incoming.items()
            }
        elif isinstance(node, Guard):
            if node.predicate:
                result["predicate"] = {
                    "kind": node.predicate.kind,
                    "negated": node.predicate.negated,
                    "args": [str(a) for a in node.predicate.args],
                }
                if node.predicate.subject:
                    result["predicate"]["subject"] = {
                        "name": node.predicate.subject.name,
                        "version": node.predicate.subject.version,
                    }
            result["true_branch"] = node.true_branch
            result["false_branch"] = node.false_branch
        elif isinstance(node, Return):
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
        elif isinstance(node, If):
            if node.test:
                result["test"] = IRSerializer._to_dict(node.test)
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["orelse"] = [IRSerializer._to_dict(s) for s in node.orelse]
        elif isinstance(node, While):
            if node.test:
                result["test"] = IRSerializer._to_dict(node.test)
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["orelse"] = [IRSerializer._to_dict(s) for s in node.orelse]
        elif isinstance(node, For):
            if node.target:
                result["target"] = {"name": node.target.name, "version": node.target.version}
            if node.iter_expr:
                result["iter"] = IRSerializer._to_dict(node.iter_expr)
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["orelse"] = [IRSerializer._to_dict(s) for s in node.orelse]
        elif isinstance(node, With):
            result["items"] = [IRSerializer._to_dict(it) for it in node.items]
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
        elif isinstance(node, Try):
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["orelse"] = [IRSerializer._to_dict(s) for s in node.orelse]
            result["finalbody"] = [IRSerializer._to_dict(s) for s in node.finalbody]
        elif isinstance(node, Raise):
            if node.exc:
                result["exc"] = IRSerializer._to_dict(node.exc)
            if node.cause:
                result["cause"] = IRSerializer._to_dict(node.cause)
        elif isinstance(node, Yield):
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
        elif isinstance(node, Await):
            if node.value:
                result["value"] = IRSerializer._to_dict(node.value)
        elif isinstance(node, FunctionDef):
            result["name"] = node.name
            result["params"] = node.params
            result["is_async"] = node.is_async
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["decorators"] = [IRSerializer._to_dict(d) for d in node.decorators]
        elif isinstance(node, ClassDef):
            result["name"] = node.name
            result["bases"] = [IRSerializer._to_dict(b) for b in node.bases]
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
            result["decorators"] = [IRSerializer._to_dict(d) for d in node.decorators]
        elif isinstance(node, Module):
            result["body"] = [IRSerializer._to_dict(s) for s in node.body]
        elif isinstance(node, Import):
            result["names"] = [[n, a] for n, a in node.names]
            result["module"] = node.module
            result["is_from"] = node.is_from
        elif isinstance(node, Lambda):
            result["args"] = node.args
            if node.body:
                result["body"] = IRSerializer._to_dict(node.body)
        elif isinstance(node, TupleExpr):
            result["elts"] = [IRSerializer._to_dict(e) for e in node.elts]
        elif isinstance(node, ListExpr):
            result["elts"] = [IRSerializer._to_dict(e) for e in node.elts]
        elif isinstance(node, DictExpr):
            result["keys"] = [IRSerializer._to_dict(k) if k else None for k in node.keys]
            result["values"] = [IRSerializer._to_dict(v) for v in node.values]
        elif isinstance(node, SetExpr):
            result["elts"] = [IRSerializer._to_dict(e) for e in node.elts]
        elif isinstance(node, Assert):
            if node.test:
                result["test"] = IRSerializer._to_dict(node.test)
            if node.msg:
                result["msg"] = IRSerializer._to_dict(node.msg)
        elif isinstance(node, Delete):
            result["targets"] = [IRSerializer._to_dict(t) for t in node.targets]
        elif isinstance(node, Global):
            result["names"] = node.names
        elif isinstance(node, Nonlocal):
            result["names"] = node.names

        return result

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> IRNode:
        node_type = data.get("type", "")
        sr = None
        sr_data = data.get("source_range")
        if sr_data:
            sr = SourceRange(**sr_data)

        if node_type == "Constant":
            node = Constant(value=data.get("value"), source_range=sr)
            te = data.get("type_expr")
            if te:
                node.type_expr = TypeExpr(te)
            return node
        elif node_type == "Name":
            v = data.get("var")
            var = SSAVar(v["name"], v["version"]) if v else None
            return Name(var=var, source_range=sr)
        elif node_type == "BinaryOp":
            left = IRSerializer._from_dict(data["left"]) if "left" in data else None
            right = IRSerializer._from_dict(data["right"]) if "right" in data else None
            return BinaryOp(op=data.get("op", ""), left=left, right=right, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "UnaryOp":
            operand = IRSerializer._from_dict(data["operand"]) if "operand" in data else None
            return UnaryOp(op=data.get("op", ""), operand=operand, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Compare":
            left = IRSerializer._from_dict(data["left"]) if "left" in data else None
            right = IRSerializer._from_dict(data["right"]) if "right" in data else None
            return Compare(op=data.get("op", ""), left=left, right=right, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Call":
            func = IRSerializer._from_dict(data["func"]) if "func" in data else None
            args = [IRSerializer._from_dict(a) for a in data.get("args", [])]
            kwargs = {k: IRSerializer._from_dict(v) for k, v in data.get("kwargs", {}).items()}
            return Call(func=func, args=args, kwargs=kwargs, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Attribute":
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            return Attribute(value=value, attr=data.get("attr", ""), source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Subscript":
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            index = IRSerializer._from_dict(data["index"]) if "index" in data else None
            return Subscript(value=value, index=index, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Assign":
            t = data.get("target")
            target = SSAVar(t["name"], t["version"]) if t else None
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            return Assign(target=target, value=value, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "PhiNode":
            d = data.get("dest")
            dest = SSAVar(d["name"], d["version"]) if d else None
            incoming = {
                k: SSAVar(v["name"], v["version"])
                for k, v in data.get("incoming", {}).items()
            }
            return PhiNode(dest=dest, incoming=incoming, source_range=sr)
        elif node_type == "Guard":
            pred_data = data.get("predicate")
            pred = None
            if pred_data:
                subj_data = pred_data.get("subject")
                subj = SSAVar(subj_data["name"], subj_data["version"]) if subj_data else None
                pred = RefinementPredicate(
                    kind=pred_data.get("kind", ""),
                    subject=subj,
                    negated=pred_data.get("negated", False),
                )
            return Guard(
                predicate=pred,
                true_branch=data.get("true_branch"),
                false_branch=data.get("false_branch"),
                source_range=sr,
            )
        elif node_type == "Return":
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            return Return(value=value, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "If":
            test = IRSerializer._from_dict(data["test"]) if "test" in data else None
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            orelse = [IRSerializer._from_dict(s) for s in data.get("orelse", [])]
            return If(test=test, body=body, orelse=orelse, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "While":
            test = IRSerializer._from_dict(data["test"]) if "test" in data else None
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            orelse = [IRSerializer._from_dict(s) for s in data.get("orelse", [])]
            return While(test=test, body=body, orelse=orelse, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "For":
            t = data.get("target")
            target = SSAVar(t["name"], t["version"]) if t else None
            iter_expr = IRSerializer._from_dict(data["iter"]) if "iter" in data else None
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            orelse = [IRSerializer._from_dict(s) for s in data.get("orelse", [])]
            return For(target=target, iter_expr=iter_expr, body=body, orelse=orelse, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "With":
            items = [IRSerializer._from_dict(it) for it in data.get("items", [])]
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            return With(items=items, body=body, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Try":
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            orelse = [IRSerializer._from_dict(s) for s in data.get("orelse", [])]
            finalbody = [IRSerializer._from_dict(s) for s in data.get("finalbody", [])]
            return Try(body=body, orelse=orelse, finalbody=finalbody, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Raise":
            exc = IRSerializer._from_dict(data["exc"]) if "exc" in data else None
            cause = IRSerializer._from_dict(data["cause"]) if "cause" in data else None
            return Raise(exc=exc, cause=cause, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Yield":
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            return Yield(value=value, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Await":
            value = IRSerializer._from_dict(data["value"]) if "value" in data else None
            return Await(value=value, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "FunctionDef":
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            decs = [IRSerializer._from_dict(d) for d in data.get("decorators", [])]
            return FunctionDef(
                name=data.get("name", ""),
                params=data.get("params", []),
                is_async=data.get("is_async", False),
                body=body,
                decorators=decs,
                source_range=sr,
            )  # type: ignore[arg-type]
        elif node_type == "ClassDef":
            bases = [IRSerializer._from_dict(b) for b in data.get("bases", [])]
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            decs = [IRSerializer._from_dict(d) for d in data.get("decorators", [])]
            return ClassDef(name=data.get("name", ""), bases=bases, body=body, decorators=decs, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Module":
            body = [IRSerializer._from_dict(s) for s in data.get("body", [])]
            return Module(body=body, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Import":
            names = [(n, a) for n, a in data.get("names", [])]
            return Import(
                names=names,
                module=data.get("module"),
                is_from=data.get("is_from", False),
                source_range=sr,
            )
        elif node_type == "Lambda":
            body = IRSerializer._from_dict(data["body"]) if "body" in data else None
            return Lambda(args=data.get("args", []), body=body, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "TupleExpr":
            elts = [IRSerializer._from_dict(e) for e in data.get("elts", [])]
            return TupleExpr(elts=elts, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "ListExpr":
            elts = [IRSerializer._from_dict(e) for e in data.get("elts", [])]
            return ListExpr(elts=elts, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "DictExpr":
            keys = [IRSerializer._from_dict(k) if k else None for k in data.get("keys", [])]
            values = [IRSerializer._from_dict(v) for v in data.get("values", [])]
            return DictExpr(keys=keys, values=values, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "SetExpr":
            elts = [IRSerializer._from_dict(e) for e in data.get("elts", [])]
            return SetExpr(elts=elts, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Assert":
            test = IRSerializer._from_dict(data["test"]) if "test" in data else None
            msg = IRSerializer._from_dict(data["msg"]) if "msg" in data else None
            return Assert(test=test, msg=msg, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Delete":
            targets = [IRSerializer._from_dict(t) for t in data.get("targets", [])]
            return Delete(targets=targets, source_range=sr)  # type: ignore[arg-type]
        elif node_type == "Global":
            return Global(names=data.get("names", []), source_range=sr)
        elif node_type == "Nonlocal":
            return Nonlocal(names=data.get("names", []), source_range=sr)
        elif node_type == "Break":
            return Break(source_range=sr)
        elif node_type == "Continue":
            return Continue(source_range=sr)
        elif node_type == "Pass":
            return Pass(source_range=sr)
        else:
            return IRNode(source_range=sr)


# ===================================================================
# IRCloner
# ===================================================================

class IRCloner:
    """Deep clone IR nodes with SSA variable renumbering."""

    def __init__(self, version_offset: int = 1000) -> None:
        self.version_offset = version_offset
        self._var_map: Dict[str, SSAVar] = {}

    def _remap_var(self, var: SSAVar) -> SSAVar:
        key = str(var)
        if key not in self._var_map:
            self._var_map[key] = SSAVar(var.name, var.version + self.version_offset)
        return self._var_map[key]

    def clone(self, node: IRNode) -> IRNode:
        """Deep clone with renumbered SSA variables."""
        self._var_map = {}
        return self._clone_node(node)

    def _clone_node(self, node: IRNode) -> IRNode:
        if isinstance(node, Constant):
            return Constant(value=node.value, type_expr=node.type_expr, source_range=copy.deepcopy(node.source_range))
        elif isinstance(node, Name):
            new_var = self._remap_var(node.var) if node.var else None
            return Name(var=new_var, source_range=copy.deepcopy(node.source_range))
        elif isinstance(node, BinaryOp):
            return BinaryOp(
                op=node.op,
                left=self._clone_node(node.left) if node.left else None,  # type: ignore[arg-type]
                right=self._clone_node(node.right) if node.right else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, UnaryOp):
            return UnaryOp(
                op=node.op,
                operand=self._clone_node(node.operand) if node.operand else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Compare):
            return Compare(
                op=node.op,
                left=self._clone_node(node.left) if node.left else None,  # type: ignore[arg-type]
                right=self._clone_node(node.right) if node.right else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Call):
            return Call(
                func=self._clone_node(node.func) if node.func else None,  # type: ignore[arg-type]
                args=[self._clone_node(a) for a in node.args],  # type: ignore[misc]
                kwargs={k: self._clone_node(v) for k, v in node.kwargs.items()},  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Attribute):
            return Attribute(
                value=self._clone_node(node.value) if node.value else None,  # type: ignore[arg-type]
                attr=node.attr,
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Subscript):
            return Subscript(
                value=self._clone_node(node.value) if node.value else None,  # type: ignore[arg-type]
                index=self._clone_node(node.index) if node.index else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Assign):
            return Assign(
                target=self._remap_var(node.target) if node.target else None,
                value=self._clone_node(node.value) if node.value else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, PhiNode):
            return PhiNode(
                dest=self._remap_var(node.dest) if node.dest else None,
                incoming={k: self._remap_var(v) for k, v in node.incoming.items()},
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Guard):
            new_pred = None
            if node.predicate:
                new_pred = RefinementPredicate(
                    kind=node.predicate.kind,
                    subject=self._remap_var(node.predicate.subject) if node.predicate.subject else None,
                    args=list(node.predicate.args),
                    negated=node.predicate.negated,
                )
            return Guard(
                predicate=new_pred,
                target=self._remap_var(node.target) if node.target else None,
                true_branch=node.true_branch,
                false_branch=node.false_branch,
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Return):
            return Return(
                value=self._clone_node(node.value) if node.value else None,  # type: ignore[arg-type]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, If):
            return If(
                test=self._clone_node(node.test) if node.test else None,  # type: ignore[arg-type]
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                orelse=[self._clone_node(s) for s in node.orelse],  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, While):
            return While(
                test=self._clone_node(node.test) if node.test else None,  # type: ignore[arg-type]
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                orelse=[self._clone_node(s) for s in node.orelse],  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, For):
            return For(
                target=self._remap_var(node.target) if node.target else None,
                iter_expr=self._clone_node(node.iter_expr) if node.iter_expr else None,  # type: ignore[arg-type]
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                orelse=[self._clone_node(s) for s in node.orelse],  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, FunctionDef):
            return FunctionDef(
                name=node.name,
                params=list(node.params),
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                decorators=[self._clone_node(d) for d in node.decorators],  # type: ignore[misc]
                is_async=node.is_async,
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, ClassDef):
            return ClassDef(
                name=node.name,
                bases=[self._clone_node(b) for b in node.bases],  # type: ignore[misc]
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                decorators=[self._clone_node(d) for d in node.decorators],  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        elif isinstance(node, Module):
            return Module(
                body=[self._clone_node(s) for s in node.body],  # type: ignore[misc]
                source_range=copy.deepcopy(node.source_range),
            )
        else:
            return copy.deepcopy(node)

    def clone_function(self, node: FunctionDef) -> FunctionDef:
        result = self._clone_node(node)
        assert isinstance(result, FunctionDef)
        return result

    def get_variable_mapping(self) -> Dict[str, SSAVar]:
        return dict(self._var_map)


# ===================================================================
# GuardExtractorVisitor
# ===================================================================

class GuardExtractorVisitor(IRVisitor):
    """Extract guards from IR and normalize to predicate templates."""

    def __init__(self) -> None:
        super().__init__()
        self.guards: List[RefinementPredicate] = []

    def extract(self, root: IRNode) -> List[RefinementPredicate]:
        self.guards = []
        self.visit(root)
        return self.guards

    def visit_Guard(self, node: Guard) -> Any:
        if node.predicate:
            self.guards.append(node.predicate)
        return self.generic_visit(node)

    def visit_Call(self, node: Call) -> Any:
        self._check_isinstance(node)
        self._check_hasattr(node)
        self._check_callable(node)
        self._check_type_call(node)
        return self.generic_visit(node)

    def visit_Compare(self, node: Compare) -> Any:
        self._check_none_compare(node)
        self._check_type_compare(node)
        self._check_numeric_compare(node)
        return self.generic_visit(node)

    def visit_Name(self, node: Name) -> Any:
        # truthiness check (e.g., `if x:`)
        if node.var:
            pred = RefinementPredicate(
                kind="truthiness",
                subject=node.var,
            )
            self.guards.append(pred)
        return None

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        if node.op == "not" and isinstance(node.operand, Name) and node.operand.var:
            pred = RefinementPredicate(
                kind="truthiness",
                subject=node.operand.var,
                negated=True,
            )
            self.guards.append(pred)
        return self.generic_visit(node)

    def _check_isinstance(self, node: Call) -> None:
        if isinstance(node.func, Name) and node.func.var and node.func.var.name == "isinstance":
            if len(node.args) == 2:
                subj_node = node.args[0]
                type_node = node.args[1]
                if isinstance(subj_node, Name) and subj_node.var:
                    pred = RefinementPredicate(
                        kind="isinstance",
                        subject=subj_node.var,
                        args=[self._extract_type_arg(type_node)],
                    )
                    self.guards.append(pred)

    def _check_hasattr(self, node: Call) -> None:
        if isinstance(node.func, Name) and node.func.var and node.func.var.name == "hasattr":
            if len(node.args) == 2:
                subj_node = node.args[0]
                attr_node = node.args[1]
                if isinstance(subj_node, Name) and subj_node.var:
                    attr_name = ""
                    if isinstance(attr_node, Constant) and isinstance(attr_node.value, str):
                        attr_name = attr_node.value
                    pred = RefinementPredicate(
                        kind="hasattr",
                        subject=subj_node.var,
                        args=[attr_name],
                    )
                    self.guards.append(pred)

    def _check_callable(self, node: Call) -> None:
        if isinstance(node.func, Name) and node.func.var and node.func.var.name == "callable":
            if len(node.args) == 1:
                subj_node = node.args[0]
                if isinstance(subj_node, Name) and subj_node.var:
                    pred = RefinementPredicate(
                        kind="callable",
                        subject=subj_node.var,
                    )
                    self.guards.append(pred)

    def _check_type_call(self, node: Call) -> None:
        if isinstance(node.func, Name) and node.func.var and node.func.var.name == "type":
            if len(node.args) == 1:
                subj_node = node.args[0]
                if isinstance(subj_node, Name) and subj_node.var:
                    pred = RefinementPredicate(
                        kind="typeof",
                        subject=subj_node.var,
                    )
                    self.guards.append(pred)

    def _check_none_compare(self, node: Compare) -> None:
        if node.op in ("is", "is not", "==", "!="):
            subj = None
            is_none = False
            if isinstance(node.left, Name) and isinstance(node.right, Constant):
                subj = node.left.var
                is_none = node.right.value is None
            elif isinstance(node.right, Name) and isinstance(node.left, Constant):
                subj = node.right.var
                is_none = node.left.value is None
            if subj and is_none:
                negated = node.op in ("is not", "!=")
                pred = RefinementPredicate(
                    kind="none_check",
                    subject=subj,
                    negated=negated,
                )
                self.guards.append(pred)

    def _check_type_compare(self, node: Compare) -> None:
        """Check for `type(x) is T` patterns."""
        if node.op in ("is", "=="):
            if isinstance(node.left, Call):
                if isinstance(node.left.func, Name) and node.left.func.var:
                    if node.left.func.var.name == "type" and len(node.left.args) == 1:
                        subj_node = node.left.args[0]
                        if isinstance(subj_node, Name) and subj_node.var:
                            pred = RefinementPredicate(
                                kind="typeof",
                                subject=subj_node.var,
                                args=[self._extract_type_arg(node.right)],
                            )
                            self.guards.append(pred)

    def _check_numeric_compare(self, node: Compare) -> None:
        """Extract comparison guards (e.g., x > 0)."""
        if node.op in ("<", ">", "<=", ">=", "==", "!="):
            if isinstance(node.left, Name) and node.left.var:
                pred = RefinementPredicate(
                    kind="comparison",
                    subject=node.left.var,
                    args=[node.op, self._expr_to_str(node.right)],
                )
                self.guards.append(pred)
            elif isinstance(node.right, Name) and node.right.var:
                # flip the comparison
                flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
                pred = RefinementPredicate(
                    kind="comparison",
                    subject=node.right.var,
                    args=[flip.get(node.op, node.op), self._expr_to_str(node.left)],
                )
                self.guards.append(pred)

    def _extract_type_arg(self, node: Optional[Expression]) -> str:
        if isinstance(node, Name) and node.var:
            return node.var.name
        if isinstance(node, Constant):
            return str(node.value)
        if isinstance(node, TupleExpr):
            return f"({', '.join(self._extract_type_arg(e) for e in node.elts)})"
        return "?"

    def _expr_to_str(self, node: Optional[Expression]) -> str:
        if node is None:
            return "?"
        if isinstance(node, Constant):
            return repr(node.value)
        if isinstance(node, Name) and node.var:
            return str(node.var)
        return "?"


# ===================================================================
# TypeAnnotationVisitor
# ===================================================================

@dataclass
class TypeAnnotationInfo:
    """Extracted type annotation."""
    name: str
    annotation: str
    kind: str  # param, return, variable
    source_range: Optional[SourceRange] = None


class TypeAnnotationVisitor(IRVisitor):
    """Extract type annotations from IR."""

    def __init__(self) -> None:
        super().__init__()
        self.annotations: List[TypeAnnotationInfo] = []

    def extract(self, root: IRNode) -> List[TypeAnnotationInfo]:
        self.annotations = []
        self.visit(root)
        return self.annotations

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        for param, ann in node.param_annotations.items():
            self.annotations.append(TypeAnnotationInfo(
                name=param,
                annotation=str(ann),
                kind="param",
                source_range=node.source_range,
            ))
        if node.return_annotation:
            self.annotations.append(TypeAnnotationInfo(
                name=node.name,
                annotation=str(node.return_annotation),
                kind="return",
                source_range=node.source_range,
            ))
        return self.generic_visit(node)

    def visit_Assign(self, node: Assign) -> Any:
        if node.target and node.metadata.get("annotation"):
            self.annotations.append(TypeAnnotationInfo(
                name=str(node.target),
                annotation=str(node.metadata["annotation"]),
                kind="variable",
                source_range=node.source_range,
            ))
        return self.generic_visit(node)


# ===================================================================
# SideEffectVisitor
# ===================================================================

class SideEffectKind(enum.Enum):
    READ_VAR = "read_var"
    WRITE_VAR = "write_var"
    READ_HEAP = "read_heap"
    WRITE_HEAP = "write_heap"
    CALL = "call"
    IO = "io"
    EXCEPTION = "exception"
    PURE = "pure"


@dataclass
class SideEffect:
    """A side effect of an IR node."""
    kind: SideEffectKind
    target: Optional[str] = None
    description: str = ""


class SideEffectVisitor(IRVisitor):
    """Analyze side effects of IR nodes."""

    IO_FUNCTIONS = frozenset({"print", "input", "open", "read", "write", "close"})
    PURE_BUILTINS = frozenset({
        "len", "abs", "min", "max", "sum", "sorted", "reversed",
        "enumerate", "zip", "map", "filter", "range", "isinstance",
        "issubclass", "hasattr", "type", "id", "hash", "repr", "str",
        "int", "float", "bool", "list", "dict", "set", "tuple",
        "frozenset", "chr", "ord", "hex", "oct", "bin",
    })

    def __init__(self) -> None:
        super().__init__()
        self.effects: List[SideEffect] = []

    def analyze(self, node: IRNode) -> List[SideEffect]:
        self.effects = []
        self.visit(node)
        return self.effects

    def is_pure(self, node: IRNode) -> bool:
        effects = self.analyze(node)
        return all(e.kind == SideEffectKind.PURE or e.kind == SideEffectKind.READ_VAR for e in effects)

    def visit_Assign(self, node: Assign) -> Any:
        if node.target:
            self.effects.append(SideEffect(
                kind=SideEffectKind.WRITE_VAR,
                target=str(node.target),
                description=f"write to {node.target}",
            ))
        if node.value:
            self.visit(node.value)
        return None

    def visit_Name(self, node: Name) -> Any:
        if node.var:
            self.effects.append(SideEffect(
                kind=SideEffectKind.READ_VAR,
                target=str(node.var),
                description=f"read {node.var}",
            ))
        return None

    def visit_Call(self, node: Call) -> Any:
        func_name = ""
        if isinstance(node.func, Name) and node.func.var:
            func_name = node.func.var.name
        elif isinstance(node.func, Attribute):
            func_name = node.func.attr

        if func_name in self.IO_FUNCTIONS:
            self.effects.append(SideEffect(
                kind=SideEffectKind.IO,
                target=func_name,
                description=f"I/O: {func_name}",
            ))
        elif func_name in self.PURE_BUILTINS:
            self.effects.append(SideEffect(
                kind=SideEffectKind.PURE,
                target=func_name,
                description=f"pure: {func_name}",
            ))
        else:
            self.effects.append(SideEffect(
                kind=SideEffectKind.CALL,
                target=func_name,
                description=f"call: {func_name}",
            ))

        if node.func:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for v in node.kwargs.values():
            self.visit(v)
        return None

    def visit_Attribute(self, node: Attribute) -> Any:
        self.effects.append(SideEffect(
            kind=SideEffectKind.READ_HEAP,
            target=node.attr,
            description=f"read attr .{node.attr}",
        ))
        if node.value:
            self.visit(node.value)
        return None

    def visit_Subscript(self, node: Subscript) -> Any:
        self.effects.append(SideEffect(
            kind=SideEffectKind.READ_HEAP,
            description="subscript read",
        ))
        if node.value:
            self.visit(node.value)
        if node.index:
            self.visit(node.index)
        return None

    def visit_Raise(self, node: Raise) -> Any:
        self.effects.append(SideEffect(
            kind=SideEffectKind.EXCEPTION,
            description="raise exception",
        ))
        return self.generic_visit(node)

    def visit_Yield(self, node: Yield) -> Any:
        self.effects.append(SideEffect(
            kind=SideEffectKind.IO,
            description="yield (generator effect)",
        ))
        return self.generic_visit(node)

    def visit_Await(self, node: Await) -> Any:
        self.effects.append(SideEffect(
            kind=SideEffectKind.IO,
            description="await (async effect)",
        ))
        return self.generic_visit(node)

    def visit_Delete(self, node: Delete) -> Any:
        for t in node.targets:
            self.effects.append(SideEffect(
                kind=SideEffectKind.WRITE_HEAP,
                description=f"delete",
            ))
        return self.generic_visit(node)


# ===================================================================
# ConstantFolder
# ===================================================================

class ConstantFolder(IRTransformer):
    """Constant folding: evaluate operations on constant operands at compile time."""

    def __init__(self) -> None:
        super().__init__()
        self.folded_count = 0

    def transform_BinaryOp(self, node: BinaryOp) -> IRNode:
        # first transform children
        new_left = self.transform(node.left) if node.left else None
        new_right = self.transform(node.right) if node.right else None

        if isinstance(new_left, Constant) and isinstance(new_right, Constant):
            result = self._eval_binary(node.op, new_left.value, new_right.value)
            if result is not _FOLD_FAIL:
                self.folded_count += 1
                return Constant(value=result, source_range=node.source_range)

        if new_left is not node.left or new_right is not node.right:
            result_node = copy.copy(node)
            result_node.left = new_left  # type: ignore[assignment]
            result_node.right = new_right  # type: ignore[assignment]
            return result_node
        return node

    def transform_UnaryOp(self, node: UnaryOp) -> IRNode:
        new_operand = self.transform(node.operand) if node.operand else None

        if isinstance(new_operand, Constant):
            result = self._eval_unary(node.op, new_operand.value)
            if result is not _FOLD_FAIL:
                self.folded_count += 1
                return Constant(value=result, source_range=node.source_range)

        if new_operand is not node.operand:
            result_node = copy.copy(node)
            result_node.operand = new_operand  # type: ignore[assignment]
            return result_node
        return node

    def transform_Compare(self, node: Compare) -> IRNode:
        new_left = self.transform(node.left) if node.left else None
        new_right = self.transform(node.right) if node.right else None

        if isinstance(new_left, Constant) and isinstance(new_right, Constant):
            result = self._eval_compare(node.op, new_left.value, new_right.value)
            if result is not _FOLD_FAIL:
                self.folded_count += 1
                return Constant(value=result, source_range=node.source_range)

        if new_left is not node.left or new_right is not node.right:
            result_node = copy.copy(node)
            result_node.left = new_left  # type: ignore[assignment]
            result_node.right = new_right  # type: ignore[assignment]
            return result_node
        return node

    def transform_If(self, node: If) -> IRNode:
        new_test = self.transform(node.test) if node.test else None
        new_body = self._transform_stmt_list(node.body)
        new_else = self._transform_stmt_list(node.orelse)

        # dead branch elimination
        if isinstance(new_test, Constant):
            if new_test.value:
                self.folded_count += 1
                if len(new_body) == 1:
                    return new_body[0]
                # return a module-like wrapper
                return Module(body=new_body, source_range=node.source_range)
            else:
                self.folded_count += 1
                if new_else:
                    if len(new_else) == 1:
                        return new_else[0]
                    return Module(body=new_else, source_range=node.source_range)
                return Pass(source_range=node.source_range)

        result = copy.copy(node)
        result.test = new_test  # type: ignore[assignment]
        result.body = new_body
        result.orelse = new_else
        return result

    def _eval_binary(self, op: str, left: Any, right: Any) -> Any:
        try:
            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                if right == 0:
                    return _FOLD_FAIL
                return left / right
            elif op == "//":
                if right == 0:
                    return _FOLD_FAIL
                return left // right
            elif op == "%":
                if right == 0:
                    return _FOLD_FAIL
                return left % right
            elif op == "**":
                return left ** right
            elif op == "&":
                return left & right
            elif op == "|":
                return left | right
            elif op == "^":
                return left ^ right
            elif op == "<<":
                return left << right
            elif op == ">>":
                return left >> right
            elif op == "and":
                return left and right
            elif op == "or":
                return left or right
        except (TypeError, ValueError, OverflowError, ArithmeticError):
            pass
        return _FOLD_FAIL

    def _eval_unary(self, op: str, operand: Any) -> Any:
        try:
            if op == "-":
                return -operand
            elif op == "+":
                return +operand
            elif op == "~":
                return ~operand
            elif op == "not":
                return not operand
        except (TypeError, ValueError):
            pass
        return _FOLD_FAIL

    def _eval_compare(self, op: str, left: Any, right: Any) -> Any:
        try:
            if op == "<":
                return left < right
            elif op == ">":
                return left > right
            elif op == "<=":
                return left <= right
            elif op == ">=":
                return left >= right
            elif op == "==":
                return left == right
            elif op == "!=":
                return left != right
            elif op == "is":
                return left is right
            elif op == "is not":
                return left is not right
            elif op == "in":
                return left in right
            elif op == "not in":
                return left not in right
        except (TypeError, ValueError):
            pass
        return _FOLD_FAIL


# sentinel for failed folding
_FOLD_FAIL = object()


# ===================================================================
# CopyPropagator
# ===================================================================

class CopyPropagator(IRTransformer):
    """Copy propagation: replace uses of x = y with y."""

    def __init__(self) -> None:
        super().__init__()
        self.copies: Dict[str, SSAVar] = {}
        self.propagated_count = 0
        self.removed_copies: List[Assign] = []

    def propagate(self, root: IRNode) -> IRNode:
        """Run copy propagation."""
        # first pass: identify copies
        self._collect_copies(root)
        # transitive closure
        self._resolve_transitive()
        # second pass: rewrite
        result = self.transform(root)
        return result

    def _collect_copies(self, node: IRNode) -> None:
        if isinstance(node, Assign):
            if isinstance(node.value, Name) and node.value.var and node.target:
                # x = y is a copy
                key = str(node.target)
                self.copies[key] = node.value.var
                self.removed_copies.append(node)
        for child in node.children():
            self._collect_copies(child)

    def _resolve_transitive(self) -> None:
        """Resolve transitive copies: if x=y and y=z, then x=z."""
        changed = True
        while changed:
            changed = False
            for key, val in list(self.copies.items()):
                val_key = str(val)
                if val_key in self.copies:
                    self.copies[key] = self.copies[val_key]
                    changed = True

    def _resolve(self, var: SSAVar) -> SSAVar:
        key = str(var)
        if key in self.copies:
            self.propagated_count += 1
            return self.copies[key]
        return var

    def transform_Name(self, node: Name) -> IRNode:
        if node.var:
            new_var = self._resolve(node.var)
            if new_var is not node.var:
                return Name(var=new_var, source_range=node.source_range)
        return node

    def transform_Assign(self, node: Assign) -> IRNode:
        # remove copy assignments (they've been propagated)
        if node in self.removed_copies and node.target:
            key = str(node.target)
            if key in self.copies:
                return Pass(source_range=node.source_range)
        return super().transform_Assign(node)

    def transform_PhiNode(self, node: PhiNode) -> IRNode:
        new_incoming: Dict[str, SSAVar] = {}
        changed = False
        for bid, var in node.incoming.items():
            new_var = self._resolve(var)
            if new_var is not var:
                changed = True
            new_incoming[bid] = new_var
        if changed:
            result = copy.copy(node)
            result.incoming = new_incoming
            return result
        return node


# ===================================================================
# DeadCodeEliminator
# ===================================================================

class DeadCodeEliminator(IRTransformer):
    """Dead code elimination: remove unused definitions and unreachable code."""

    def __init__(self) -> None:
        super().__init__()
        self.eliminated_count = 0
        self._used_vars: Set[str] = set()
        self._side_effect_visitor = SideEffectVisitor()

    def eliminate(self, root: IRNode) -> IRNode:
        """Run dead code elimination."""
        # collect all used variables
        self._collect_uses(root)
        # transform: remove dead assignments
        result = self.transform(root)
        return result

    def _collect_uses(self, node: IRNode) -> None:
        if isinstance(node, Name) and node.var:
            self._used_vars.add(str(node.var))
        elif isinstance(node, PhiNode):
            for v in node.incoming.values():
                self._used_vars.add(str(v))
        elif isinstance(node, Guard) and node.predicate and node.predicate.subject:
            self._used_vars.add(str(node.predicate.subject))
        for child in node.children():
            self._collect_uses(child)

    def _has_side_effects(self, node: IRNode) -> bool:
        effects = self._side_effect_visitor.analyze(node)
        for e in effects:
            if e.kind not in (SideEffectKind.PURE, SideEffectKind.READ_VAR):
                return True
        return False

    def transform_Assign(self, node: Assign) -> IRNode:
        if node.target:
            key = str(node.target)
            if key not in self._used_vars:
                # check if value has side effects
                if node.value and self._has_side_effects(node.value):
                    return super().transform_Assign(node)
                self.eliminated_count += 1
                return Pass(source_range=node.source_range)
        return super().transform_Assign(node)

    def transform_PhiNode(self, node: PhiNode) -> IRNode:
        if node.dest:
            key = str(node.dest)
            if key not in self._used_vars:
                self.eliminated_count += 1
                return Pass(source_range=node.source_range)
        return node

    def transform_If(self, node: If) -> IRNode:
        new_test = self.transform(node.test) if node.test else None
        new_body = self._transform_and_filter(node.body)
        new_else = self._transform_and_filter(node.orelse)

        if not new_body and not new_else:
            if new_test and not self._has_side_effects(new_test):
                self.eliminated_count += 1
                return Pass(source_range=node.source_range)

        result = copy.copy(node)
        result.test = new_test  # type: ignore[assignment]
        result.body = new_body
        result.orelse = new_else
        return result

    def _transform_and_filter(self, stmts: List[Statement]) -> List[Statement]:
        result: List[Statement] = []
        for s in stmts:
            t = self.transform(s)
            if isinstance(t, Pass):
                continue
            if isinstance(t, Statement):
                result.append(t)
        return result

    def transform_Module(self, node: Module) -> IRNode:
        new_body = self._transform_and_filter(node.body)
        result = copy.copy(node)
        result.body = new_body
        return result

    def transform_FunctionDef(self, node: FunctionDef) -> IRNode:
        new_body = self._transform_and_filter(node.body)
        result = copy.copy(node)
        result.body = new_body if new_body else [Pass()]
        result.decorators = self._transform_expr_list(node.decorators)
        return result


# ===================================================================
# Additional utility visitors
# ===================================================================

class VariableCollector(IRVisitor):
    """Collect all SSA variables referenced in an IR tree."""

    def __init__(self) -> None:
        super().__init__()
        self.defined: Set[str] = set()
        self.used: Set[str] = set()

    def collect(self, root: IRNode) -> Tuple[Set[str], Set[str]]:
        self.defined = set()
        self.used = set()
        self.visit(root)
        return self.defined, self.used

    def visit_Assign(self, node: Assign) -> Any:
        if node.target:
            self.defined.add(str(node.target))
        return self.generic_visit(node)

    def visit_PhiNode(self, node: PhiNode) -> Any:
        if node.dest:
            self.defined.add(str(node.dest))
        for v in node.incoming.values():
            self.used.add(str(v))
        return None

    def visit_Name(self, node: Name) -> Any:
        if node.var:
            self.used.add(str(node.var))
        return None

    def visit_For(self, node: For) -> Any:
        if node.target:
            self.defined.add(str(node.target))
        return self.generic_visit(node)


class NodeCounter(IRVisitor):
    """Count nodes in an IR tree by type."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Dict[str, int] = {}
        self.total: int = 0

    def count(self, root: IRNode) -> Dict[str, int]:
        self.counts = {}
        self.total = 0
        self.visit(root)
        return self.counts

    def generic_visit(self, node: IRNode) -> Any:
        name = type(node).__name__
        self.counts[name] = self.counts.get(name, 0) + 1
        self.total += 1
        for child in node.children():
            self.visit(child)
        return None


class FreeVariableCollector(IRVisitor):
    """Collect free variables (used but not defined in scope)."""

    def __init__(self) -> None:
        super().__init__()
        self._defined: Set[str] = set()
        self._used: Set[str] = set()
        self.free: Set[str] = set()

    def collect(self, root: IRNode) -> Set[str]:
        self._defined = set()
        self._used = set()
        self.visit(root)
        self.free = self._used - self._defined
        return self.free

    def visit_Assign(self, node: Assign) -> Any:
        if node.value:
            self.visit(node.value)
        if node.target:
            self._defined.add(str(node.target))
        return None

    def visit_PhiNode(self, node: PhiNode) -> Any:
        if node.dest:
            self._defined.add(str(node.dest))
        for v in node.incoming.values():
            self._used.add(str(v))
        return None

    def visit_Name(self, node: Name) -> Any:
        if node.var:
            self._used.add(str(node.var))
        return None

    def visit_For(self, node: For) -> Any:
        if node.target:
            self._defined.add(str(node.target))
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        for p in node.params:
            self._defined.add(p)
        return self.generic_visit(node)


class ComplexityAnalyzer(IRVisitor):
    """Analyze IR complexity metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.assignments = 0
        self.branches = 0
        self.loops = 0
        self.calls = 0
        self.max_depth = 0
        self._current_depth = 0

    def analyze(self, root: IRNode) -> Dict[str, int]:
        self.assignments = 0
        self.branches = 0
        self.loops = 0
        self.calls = 0
        self.max_depth = 0
        self._current_depth = 0
        self.visit(root)
        return {
            "assignments": self.assignments,
            "branches": self.branches,
            "loops": self.loops,
            "calls": self.calls,
            "max_depth": self.max_depth,
        }

    def visit_Assign(self, node: Assign) -> Any:
        self.assignments += 1
        return self.generic_visit(node)

    def visit_If(self, node: If) -> Any:
        self.branches += 1
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        result = self.generic_visit(node)
        self._current_depth -= 1
        return result

    def visit_While(self, node: While) -> Any:
        self.loops += 1
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        result = self.generic_visit(node)
        self._current_depth -= 1
        return result

    def visit_For(self, node: For) -> Any:
        self.loops += 1
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        result = self.generic_visit(node)
        self._current_depth -= 1
        return result

    def visit_Call(self, node: Call) -> Any:
        self.calls += 1
        return self.generic_visit(node)


# ===================================================================
# Pattern matching visitor
# ===================================================================

class PatternMatcher(IRVisitor):
    """Match IR patterns for rewriting."""

    @dataclass
    class Match:
        node: IRNode
        bindings: Dict[str, IRNode] = field(default_factory=dict)

    def __init__(self) -> None:
        super().__init__()
        self.matches: List[PatternMatcher.Match] = []

    def find_isinstance_guards(self, root: IRNode) -> List[Match]:
        """Find isinstance check patterns."""
        self.matches = []

        def pred(n: IRNode) -> bool:
            if isinstance(n, Call):
                if isinstance(n.func, Name) and n.func.var and n.func.var.name == "isinstance":
                    return True
            return False

        collector = IRCollector()
        nodes = collector.collect_all(root, pred)
        for n in nodes:
            assert isinstance(n, Call)
            bindings: Dict[str, IRNode] = {}
            if len(n.args) >= 2:
                bindings["subject"] = n.args[0]
                bindings["type"] = n.args[1]
            self.matches.append(PatternMatcher.Match(node=n, bindings=bindings))
        return self.matches

    def find_none_checks(self, root: IRNode) -> List[Match]:
        """Find `x is None` / `x is not None` patterns."""
        self.matches = []

        def pred(n: IRNode) -> bool:
            if isinstance(n, Compare) and n.op in ("is", "is not"):
                if isinstance(n.right, Constant) and n.right.value is None:
                    return True
                if isinstance(n.left, Constant) and n.left.value is None:
                    return True
            return False

        collector = IRCollector()
        nodes = collector.collect_all(root, pred)
        for n in nodes:
            assert isinstance(n, Compare)
            bindings: Dict[str, IRNode] = {"compare": n}
            if isinstance(n.left, Name):
                bindings["subject"] = n.left
            elif isinstance(n.right, Name):
                bindings["subject"] = n.right
            self.matches.append(PatternMatcher.Match(node=n, bindings=bindings))
        return self.matches

    def find_attribute_accesses(self, root: IRNode) -> List[Match]:
        """Find attribute access patterns."""
        self.matches = []
        collector = IRCollector()
        nodes = collector.collect_by_type(root, Attribute)
        for n in nodes:
            assert isinstance(n, Attribute)
            bindings: Dict[str, IRNode] = {
                "value": n.value if n.value else IRNode(),
                "attr_name": Constant(value=n.attr),
            }
            self.matches.append(PatternMatcher.Match(node=n, bindings=bindings))
        return self.matches

    def find_assignments_to(self, root: IRNode, var_name: str) -> List[Match]:
        """Find all assignments to a specific variable name."""
        self.matches = []

        def pred(n: IRNode) -> bool:
            if isinstance(n, Assign) and n.target and n.target.name == var_name:
                return True
            return False

        collector = IRCollector()
        nodes = collector.collect_all(root, pred)
        for n in nodes:
            self.matches.append(PatternMatcher.Match(node=n))
        return self.matches


# ===================================================================
# InstructionSimplifier: combines multiple optimizations
# ===================================================================

class InstructionSimplifier:
    """Run multiple optimization passes on IR."""

    def __init__(self) -> None:
        self.constant_folder = ConstantFolder()
        self.copy_propagator = CopyPropagator()
        self.dead_code_eliminator = DeadCodeEliminator()

    def simplify(self, root: IRNode, max_iterations: int = 10) -> Tuple[IRNode, Dict[str, int]]:
        """Run simplification passes until fixed point."""
        stats: Dict[str, int] = {
            "constant_folds": 0,
            "copy_propagations": 0,
            "dead_eliminations": 0,
            "iterations": 0,
        }

        current = root
        for i in range(max_iterations):
            stats["iterations"] = i + 1

            self.constant_folder.folded_count = 0
            current = self.constant_folder.transform(current)
            stats["constant_folds"] += self.constant_folder.folded_count

            self.copy_propagator = CopyPropagator()
            current = self.copy_propagator.propagate(current)
            stats["copy_propagations"] += self.copy_propagator.propagated_count

            self.dead_code_eliminator = DeadCodeEliminator()
            current = self.dead_code_eliminator.eliminate(current)
            stats["dead_eliminations"] += self.dead_code_eliminator.eliminated_count

            total_changes = (
                self.constant_folder.folded_count
                + self.copy_propagator.propagated_count
                + self.dead_code_eliminator.eliminated_count
            )
            if total_changes == 0:
                break

        return current, stats


# ===================================================================
# IRDiffer: diff two IR trees
# ===================================================================

@dataclass
class IRDiffEntry:
    kind: str  # added, removed, modified
    path: str
    old_node: Optional[IRNode] = None
    new_node: Optional[IRNode] = None

    def __str__(self) -> str:
        return f"[{self.kind}] {self.path}"


class IRDiffer:
    """Diff two IR trees."""

    @staticmethod
    def diff(old: IRNode, new: IRNode, path: str = "") -> List[IRDiffEntry]:
        entries: List[IRDiffEntry] = []
        IRDiffer._diff_nodes(old, new, path or type(old).__name__, entries)
        return entries

    @staticmethod
    def _diff_nodes(old: IRNode, new: IRNode, path: str, entries: List[IRDiffEntry]) -> None:
        if type(old) != type(new):
            entries.append(IRDiffEntry(kind="modified", path=path, old_node=old, new_node=new))
            return

        # compare specific fields
        if isinstance(old, Constant) and isinstance(new, Constant):
            if old.value != new.value:
                entries.append(IRDiffEntry(kind="modified", path=f"{path}.value", old_node=old, new_node=new))
        elif isinstance(old, Name) and isinstance(new, Name):
            if old.var != new.var:
                entries.append(IRDiffEntry(kind="modified", path=f"{path}.var", old_node=old, new_node=new))
        elif isinstance(old, BinaryOp) and isinstance(new, BinaryOp):
            if old.op != new.op:
                entries.append(IRDiffEntry(kind="modified", path=f"{path}.op", old_node=old, new_node=new))

        old_children = old.children()
        new_children = new.children()

        min_len = min(len(old_children), len(new_children))
        for i in range(min_len):
            IRDiffer._diff_nodes(old_children[i], new_children[i], f"{path}[{i}]", entries)

        for i in range(min_len, len(old_children)):
            entries.append(IRDiffEntry(kind="removed", path=f"{path}[{i}]", old_node=old_children[i]))

        for i in range(min_len, len(new_children)):
            entries.append(IRDiffEntry(kind="added", path=f"{path}[{i}]", new_node=new_children[i]))
