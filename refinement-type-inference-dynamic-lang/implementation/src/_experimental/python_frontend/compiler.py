"""
Python SSA Compiler — transforms Python ASTs into SSA-based IR.

Handles Python 3.8–3.12 syntax including:
  - Full statement coverage (if/for/while/try/with/match/etc.)
  - Expression compilation with short-circuit, chained comparisons, walrus
  - Comprehension desugaring, decorator wrapping, context managers
  - SSA variable versioning with phi-node insertion at join points
  - Truthiness coercion, exception flow modeling, async/await
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
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

from src.ir.nodes import (
    AssertNode,
    AssignNode,
    AwaitNode,
    BinOp,
    BinOpNode,
    BranchNode,
    CallNode,
    ClosureCaptureNode,
    CompareNode,
    CompareOp,
    ContainerCreateNode,
    ContainerKind,
    DeleteNode,
    ExceptHandlerNode,
    FormatStringNode,
    GatedPhiNode,
    GuardKind,
    GuardNode,
    HasAttrNode,
    IRNode,
    ImportNode,
    IndexNode,
    JumpNode,
    LenNode,
    LiteralKind,
    LiteralNode,
    LoadAttrNode,
    NullCheckNode,
    PhiNode,
    RaiseNode,
    ReturnNode,
    SliceNode,
    SourceLocation,
    SSAVar,
    StoreAttrNode,
    StoreIndexNode,
    TruthinessNode,
    TypeAnnotation,
    TypeNarrowNode,
    TypeTestNode,
    UnaryOp,
    UnaryOpNode,
    UnpackNode,
    YieldNode,
)

# ---------------------------------------------------------------------------
# IR container types (forward-compatible with src.ir.unified)
# ---------------------------------------------------------------------------


@dataclass
class IRBasicBlock:
    """A basic block in the control-flow graph."""

    label: str
    nodes: List[IRNode] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    def append(self, node: IRNode) -> None:
        self.nodes.append(node)

    @property
    def terminator(self) -> Optional[IRNode]:
        if self.nodes and isinstance(
            self.nodes[-1], (BranchNode, JumpNode, ReturnNode, RaiseNode)
        ):
            return self.nodes[-1]
        return None

    def pretty_print(self, indent: int = 0) -> str:
        pfx = "  " * indent
        lines = [f"{pfx}{self.label}:"]
        lines.append(f"{pfx}  ; preds: {', '.join(self.predecessors)}")
        for n in self.nodes:
            lines.append(n.pretty_print(indent + 1))
        return "\n".join(lines)


@dataclass
class CFG:
    """Control-flow graph: an ordered mapping of block labels to blocks."""

    blocks: Dict[str, IRBasicBlock] = field(default_factory=dict)
    entry: str = "entry"

    def add_block(self, block: IRBasicBlock) -> None:
        self.blocks[block.label] = block

    def get_block(self, label: str) -> IRBasicBlock:
        return self.blocks[label]

    def add_edge(self, src: str, dst: str) -> None:
        sb = self.blocks[src]
        db = self.blocks[dst]
        if dst not in sb.successors:
            sb.successors.append(dst)
        if src not in db.predecessors:
            db.predecessors.append(src)

    @property
    def block_order(self) -> List[str]:
        visited: List[str] = []
        seen: Set[str] = set()
        stack = [self.entry]
        while stack:
            lbl = stack.pop()
            if lbl in seen or lbl not in self.blocks:
                continue
            seen.add(lbl)
            visited.append(lbl)
            for s in reversed(self.blocks[lbl].successors):
                if s not in seen:
                    stack.append(s)
        return visited

    def pretty_print(self) -> str:
        return "\n\n".join(
            self.blocks[lbl].pretty_print()
            for lbl in self.block_order
            if lbl in self.blocks
        )


@dataclass
class IRParameter:
    """Function parameter descriptor."""

    name: str
    annotation: Optional[str] = None
    default_var: Optional[SSAVar] = None
    is_star: bool = False
    is_double_star: bool = False


@dataclass
class IRFunction:
    """A compiled function or method."""

    name: str
    params: List[IRParameter] = field(default_factory=list)
    cfg: CFG = field(default_factory=CFG)
    return_annotation: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    closure_captures: List[SSAVar] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    source_file: str = "<unknown>"
    source_line: int = 0

    def pretty_print(self) -> str:
        prefix = "async " if self.is_async else ""
        params = ", ".join(p.name for p in self.params)
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        header = f"{prefix}def {self.name}({params}){ret}:"
        return f"{header}\n{self.cfg.pretty_print()}"


@dataclass
class IRClass:
    """A compiled class."""

    name: str
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, IRFunction] = field(default_factory=dict)
    class_vars: Dict[str, SSAVar] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    source_file: str = "<unknown>"
    source_line: int = 0


@dataclass
class IRModule:
    """Top-level compilation unit."""

    name: str = "<module>"
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    classes: Dict[str, IRClass] = field(default_factory=dict)
    top_level_cfg: CFG = field(default_factory=CFG)
    imports: List[ImportNode] = field(default_factory=list)
    source_file: str = "<unknown>"

    def pretty_print(self) -> str:
        parts: List[str] = [f"module {self.name}:"]
        parts.append(self.top_level_cfg.pretty_print())
        for fn in self.functions.values():
            parts.append("")
            parts.append(fn.pretty_print())
        for cls in self.classes.values():
            parts.append(f"\nclass {cls.name}({', '.join(cls.bases)}):")
            for m in cls.methods.values():
                parts.append(m.pretty_print())
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scope resolution (LEGB)
# ---------------------------------------------------------------------------

class _ScopeKind(Enum):
    MODULE = auto()
    FUNCTION = auto()
    CLASS = auto()
    COMPREHENSION = auto()


@dataclass
class _Scope:
    kind: _ScopeKind
    name: str
    local_vars: Set[str] = field(default_factory=set)
    global_vars: Set[str] = field(default_factory=set)
    nonlocal_vars: Set[str] = field(default_factory=set)
    free_vars: Set[str] = field(default_factory=set)


class ScopeResolver(ast.NodeVisitor):
    """Resolves Python LEGB scoping, nonlocal/global declarations, and
    closure (free) variables by pre-scanning the AST."""

    def __init__(self) -> None:
        self._scopes: List[_Scope] = []
        self.scope_map: Dict[int, _Scope] = {}

    @property
    def _current(self) -> _Scope:
        return self._scopes[-1]

    def _push(self, kind: _ScopeKind, name: str, node_id: int) -> _Scope:
        scope = _Scope(kind=kind, name=name)
        self._scopes.append(scope)
        self.scope_map[node_id] = scope
        return scope

    def _pop(self) -> _Scope:
        return self._scopes.pop()

    def resolve(self, tree: ast.Module) -> Dict[int, _Scope]:
        self._push(_ScopeKind.MODULE, "<module>", id(tree))
        self.generic_visit(tree)
        self._pop()
        self._compute_free_vars()
        return self.scope_map

    # -- declarations --------------------------------------------------------

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._current.global_vars.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._current.nonlocal_vars.add(name)

    # -- binding forms -------------------------------------------------------

    def _record_assign(self, name: str) -> None:
        sc = self._current
        if name not in sc.global_vars and name not in sc.nonlocal_vars:
            sc.local_vars.add(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._record_assign(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_assign(node.name)
        self._push(_ScopeKind.FUNCTION, node.name, id(node))
        for arg in ast.walk(node.args):
            if isinstance(arg, ast.arg):
                self._record_assign(arg.arg)
        for child in ast.iter_child_nodes(node):
            if child is not node.args:
                self.visit(child)
        self._pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_assign(node.name)
        self._push(_ScopeKind.CLASS, node.name, id(node))
        self.generic_visit(node)
        self._pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self._record_assign(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._record_assign(name)

    def visit_For(self, node: ast.For) -> None:
        self._visit_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._record_assign(node.name)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.target, ast.Name):
            self._record_assign(node.target.id)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _visit_comprehension(self, node: ast.AST) -> None:
        self._push(_ScopeKind.COMPREHENSION, "<comp>", id(node))
        self.generic_visit(node)
        self._pop()

    def _visit_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._record_assign(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_target(elt)
        elif isinstance(target, ast.Starred):
            self._visit_target(target.value)

    # -- free variable computation -------------------------------------------

    def _compute_free_vars(self) -> None:
        """Mark free variables for each scope by walking the scope chain."""
        all_scopes = list(self.scope_map.values())
        for scope in all_scopes:
            for name in scope.nonlocal_vars:
                scope.free_vars.add(name)


# ---------------------------------------------------------------------------
# SSA naming / versioning
# ---------------------------------------------------------------------------

class _SSANamer:
    """Manages SSA variable versioning with a renaming stack."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)
        self._stacks: Dict[str, List[int]] = defaultdict(lambda: [0])
        self._temp_counter: int = 0

    def fresh_version(self, name: str) -> SSAVar:
        self._counters[name] += 1
        ver = self._counters[name]
        self._stacks[name].append(ver)
        return SSAVar(name, ver)

    def current(self, name: str) -> SSAVar:
        if name in self._stacks and self._stacks[name]:
            return SSAVar(name, self._stacks[name][-1])
        return SSAVar(name, 0)

    def push_version(self, name: str, version: int) -> None:
        self._stacks[name].append(version)

    def pop_version(self, name: str) -> None:
        if name in self._stacks and len(self._stacks[name]) > 1:
            self._stacks[name].pop()

    def fresh_temp(self, prefix: str = "_t") -> SSAVar:
        self._temp_counter += 1
        name = f"{prefix}{self._temp_counter}"
        return self.fresh_version(name)

    def snapshot(self) -> Dict[str, int]:
        return {k: v[-1] for k, v in self._stacks.items() if v}

    def restore_snapshot(self, snap: Dict[str, int]) -> None:
        for name, ver in snap.items():
            self._stacks[name].append(ver)

    def pop_snapshot(self, snap: Dict[str, int]) -> None:
        for name in snap:
            self.pop_version(name)


# ---------------------------------------------------------------------------
# Builtin signatures
# ---------------------------------------------------------------------------

@dataclass
class BuiltinSig:
    """Describes a known builtin's parameter count and return type hint."""

    name: str
    min_args: int = 0
    max_args: int = -1
    return_type: str = "unknown"
    is_type_guard: bool = False


class BuiltinSignatures:
    """Known signatures for Python builtins used during compilation."""

    _SIGS: Dict[str, BuiltinSig] = {
        "len": BuiltinSig("len", 1, 1, "int"),
        "range": BuiltinSig("range", 1, 3, "range"),
        "isinstance": BuiltinSig("isinstance", 2, 2, "bool", is_type_guard=True),
        "issubclass": BuiltinSig("issubclass", 2, 2, "bool"),
        "type": BuiltinSig("type", 1, 3, "type"),
        "hasattr": BuiltinSig("hasattr", 2, 2, "bool", is_type_guard=True),
        "getattr": BuiltinSig("getattr", 2, 3, "unknown"),
        "setattr": BuiltinSig("setattr", 3, 3, "None"),
        "delattr": BuiltinSig("delattr", 2, 2, "None"),
        "print": BuiltinSig("print", 0, -1, "None"),
        "input": BuiltinSig("input", 0, 1, "str"),
        "int": BuiltinSig("int", 0, 2, "int"),
        "float": BuiltinSig("float", 0, 1, "float"),
        "str": BuiltinSig("str", 0, 1, "str"),
        "bool": BuiltinSig("bool", 0, 1, "bool"),
        "list": BuiltinSig("list", 0, 1, "list"),
        "dict": BuiltinSig("dict", 0, -1, "dict"),
        "set": BuiltinSig("set", 0, 1, "set"),
        "tuple": BuiltinSig("tuple", 0, 1, "tuple"),
        "abs": BuiltinSig("abs", 1, 1, "number"),
        "min": BuiltinSig("min", 1, -1, "unknown"),
        "max": BuiltinSig("max", 1, -1, "unknown"),
        "sum": BuiltinSig("sum", 1, 2, "number"),
        "sorted": BuiltinSig("sorted", 1, 1, "list"),
        "reversed": BuiltinSig("reversed", 1, 1, "iterator"),
        "enumerate": BuiltinSig("enumerate", 1, 2, "iterator"),
        "zip": BuiltinSig("zip", 0, -1, "iterator"),
        "map": BuiltinSig("map", 2, -1, "iterator"),
        "filter": BuiltinSig("filter", 2, 2, "iterator"),
        "any": BuiltinSig("any", 1, 1, "bool"),
        "all": BuiltinSig("all", 1, 1, "bool"),
        "iter": BuiltinSig("iter", 1, 2, "iterator"),
        "next": BuiltinSig("next", 1, 2, "unknown"),
        "id": BuiltinSig("id", 1, 1, "int"),
        "hash": BuiltinSig("hash", 1, 1, "int"),
        "repr": BuiltinSig("repr", 1, 1, "str"),
        "callable": BuiltinSig("callable", 1, 1, "bool", is_type_guard=True),
        "chr": BuiltinSig("chr", 1, 1, "str"),
        "ord": BuiltinSig("ord", 1, 1, "int"),
        "hex": BuiltinSig("hex", 1, 1, "str"),
        "oct": BuiltinSig("oct", 1, 1, "str"),
        "bin": BuiltinSig("bin", 1, 1, "str"),
        "format": BuiltinSig("format", 1, 2, "str"),
        "vars": BuiltinSig("vars", 0, 1, "dict"),
        "dir": BuiltinSig("dir", 0, 1, "list"),
        "open": BuiltinSig("open", 1, -1, "IO"),
        "super": BuiltinSig("super", 0, 2, "super"),
        "property": BuiltinSig("property", 0, 4, "property"),
        "staticmethod": BuiltinSig("staticmethod", 1, 1, "staticmethod"),
        "classmethod": BuiltinSig("classmethod", 1, 1, "classmethod"),
        "object": BuiltinSig("object", 0, 0, "object"),
    }

    @classmethod
    def get(cls, name: str) -> Optional[BuiltinSig]:
        return cls._SIGS.get(name)

    @classmethod
    def is_builtin(cls, name: str) -> bool:
        return name in cls._SIGS

    @classmethod
    def is_type_guard(cls, name: str) -> bool:
        sig = cls._SIGS.get(name)
        return sig is not None and sig.is_type_guard


# ---------------------------------------------------------------------------
# Truthiness modeler
# ---------------------------------------------------------------------------

class TruthinessModeler:
    """Inserts explicit TruthinessNode where Python's bool() semantics
    require coercion (if/while conditions, boolean operators)."""

    def __init__(self, namer: _SSANamer, loc_fn: Callable[..., SourceLocation]) -> None:
        self._namer = namer
        self._loc = loc_fn

    def coerce(
        self,
        operand: SSAVar,
        block: IRBasicBlock,
        ast_node: Optional[ast.AST] = None,
    ) -> SSAVar:
        dst = self._namer.fresh_temp("_bool")
        loc = self._loc(ast_node) if ast_node else SourceLocation()
        block.append(TruthinessNode(dst=dst, operand=operand, source_loc=loc))
        return dst


# ---------------------------------------------------------------------------
# Exception flow handler
# ---------------------------------------------------------------------------

class ExceptionFlowHandler:
    """Models try/except/finally as gated phi nodes for exception paths."""

    def __init__(self, builder: "SSABuilder") -> None:
        self._builder = builder

    def compile_try(
        self,
        node: ast.Try,
        loc: SourceLocation,
    ) -> None:
        b = self._builder
        namer = b._namer

        try_block = b._new_block("try_body")
        handler_blocks: List[str] = []
        finally_block: Optional[str] = None
        else_block: Optional[str] = None
        merge_block = b._new_block("try_merge")

        if node.finalbody:
            finally_block = b._new_block("try_finally")
        if node.orelse:
            else_block = b._new_block("try_else")

        for handler in node.handlers:
            hlabel = b._new_block("except")
            handler_blocks.append(hlabel)

        snap_before = namer.snapshot()

        # try body
        b._seal_and_jump(try_block)
        b._current_block = try_block
        b._compile_body(node.body)
        try_end = b._current_block_label

        # if no exception: go to else or finally or merge
        if else_block:
            b._seal_and_jump(else_block)
        elif finally_block:
            b._seal_and_jump(finally_block)
        else:
            b._seal_and_jump(merge_block)

        snap_after_try = namer.snapshot()

        # except handlers
        handler_end_labels: List[str] = []
        handler_snapshots: List[Dict[str, int]] = []
        for i, handler in enumerate(node.handlers):
            namer.restore_snapshot(snap_before)
            b._current_block = handler_blocks[i]
            b._ensure_block(handler_blocks[i])

            exc_types: List[str] = []
            if handler.type:
                exc_types = [b._annotation_str(handler.type)]

            exc_var: Optional[SSAVar] = None
            if handler.name:
                exc_var = namer.fresh_version(handler.name)

            b._emit(ExceptHandlerNode(
                exc_var=exc_var,
                exc_types=exc_types,
                handler_block=handler_blocks[i],
                source_loc=b._loc(handler),
            ))

            b._compile_body(handler.body)
            handler_end_labels.append(b._current_block_label)
            handler_snapshots.append(namer.snapshot())

            if finally_block:
                b._seal_and_jump(finally_block)
            else:
                b._seal_and_jump(merge_block)

        # else block
        if else_block:
            namer.restore_snapshot(snap_after_try)
            b._current_block = else_block
            b._ensure_block(else_block)
            b._compile_body(node.orelse)
            if finally_block:
                b._seal_and_jump(finally_block)
            else:
                b._seal_and_jump(merge_block)

        # finally block
        if finally_block:
            b._current_block = finally_block
            b._ensure_block(finally_block)
            b._compile_body(node.finalbody)
            b._seal_and_jump(merge_block)

        # merge
        b._current_block = merge_block
        b._ensure_block(merge_block)

        # insert phi nodes for variables modified in any branch
        all_modified: Set[str] = set()
        for snap in [snap_after_try] + handler_snapshots:
            all_modified.update(snap.keys())

        preds = [try_end] + handler_end_labels
        if else_block:
            preds.append(else_block)
        if finally_block:
            preds = [finally_block]

        for var_name in all_modified:
            if var_name.startswith("_t") or var_name.startswith("_bool"):
                continue
            incoming: Dict[str, SSAVar] = {}
            for pred in preds:
                incoming[pred] = namer.current(var_name)
            if len(set(str(v) for v in incoming.values())) > 1:
                phi_dst = namer.fresh_version(var_name)
                b._emit(PhiNode(dst=phi_dst, incoming=incoming, source_loc=loc))


# ---------------------------------------------------------------------------
# Async desugarer
# ---------------------------------------------------------------------------

class AsyncDesugarer:
    """Transforms async/await constructs into coroutine IR nodes."""

    def __init__(self, builder: "SSABuilder") -> None:
        self._builder = builder

    def compile_await(self, node: ast.Await) -> SSAVar:
        b = self._builder
        value_var = b._compile_expr(node.value)
        dst = b._namer.fresh_temp("_await")
        b._emit(AwaitNode(dst=dst, value=value_var, source_loc=b._loc(node)))
        return dst

    def compile_async_for(self, node: ast.AsyncFor) -> None:
        b = self._builder
        iter_var = b._compile_expr(node.iter)
        aiter_var = b._namer.fresh_temp("_aiter")
        b._emit(CallNode(
            result=aiter_var,
            func=SSAVar("__aiter__"),
            args=[iter_var],
            source_loc=b._loc(node),
        ))
        b._compile_for_common(node, aiter_var, is_async=True)

    def compile_async_with(self, node: ast.AsyncWith) -> None:
        b = self._builder
        b._compile_with_common(node, is_async=True)


# ---------------------------------------------------------------------------
# Desugarer — transforms complex constructs into simpler IR
# ---------------------------------------------------------------------------

class Desugarer:
    """Desugars complex Python constructs into simpler IR sequences."""

    def __init__(self, builder: "SSABuilder") -> None:
        self._builder = builder

    # -- comprehensions → explicit loops ------------------------------------

    def desugar_listcomp(self, node: ast.ListComp) -> SSAVar:
        return self._desugar_comp(node, ContainerKind.LIST, node.elt)

    def desugar_setcomp(self, node: ast.SetComp) -> SSAVar:
        return self._desugar_comp(node, ContainerKind.SET, node.elt)

    def desugar_dictcomp(self, node: ast.DictComp) -> SSAVar:
        return self._desugar_comp_dict(node)

    def desugar_genexp(self, node: ast.GeneratorExp) -> SSAVar:
        return self._desugar_comp(node, ContainerKind.LIST, node.elt)

    def _desugar_comp(
        self,
        node: ast.AST,
        kind: ContainerKind,
        elt: ast.expr,
    ) -> SSAVar:
        b = self._builder
        namer = b._namer
        loc = b._loc(node)

        accum = namer.fresh_temp("_comp")
        b._emit(ContainerCreateNode(dst=accum, kind=kind, source_loc=loc))

        generators = getattr(node, "generators", [])
        self._compile_generators(generators, 0, elt, accum, kind, loc)
        return accum

    def _desugar_comp_dict(self, node: ast.DictComp) -> SSAVar:
        b = self._builder
        namer = b._namer
        loc = b._loc(node)

        accum = namer.fresh_temp("_dcomp")
        b._emit(ContainerCreateNode(
            dst=accum, kind=ContainerKind.DICT, source_loc=loc,
        ))

        def emit_pair() -> None:
            key_var = b._compile_expr(node.key)
            val_var = b._compile_expr(node.value)
            b._emit(StoreIndexNode(
                obj=accum, index=key_var, value=val_var, source_loc=loc,
            ))

        self._compile_generators_body(node.generators, 0, emit_pair, loc)
        return accum

    def _compile_generators(
        self,
        generators: List[ast.comprehension],
        idx: int,
        elt: ast.expr,
        accum: SSAVar,
        kind: ContainerKind,
        loc: SourceLocation,
    ) -> None:
        b = self._builder

        def emit_append() -> None:
            val = b._compile_expr(elt)
            append_fn = b._namer.fresh_temp("_append")
            b._emit(LoadAttrNode(
                dst=append_fn, obj=accum, attr="append", source_loc=loc,
            ))
            result = b._namer.fresh_temp("_")
            b._emit(CallNode(
                result=result, func=append_fn, args=[val], source_loc=loc,
            ))

        self._compile_generators_body(generators, idx, emit_append, loc)

    def _compile_generators_body(
        self,
        generators: List[ast.comprehension],
        idx: int,
        body_fn: Callable[[], None],
        loc: SourceLocation,
    ) -> None:
        if idx >= len(generators):
            body_fn()
            return

        b = self._builder
        namer = b._namer
        gen = generators[idx]

        iter_var = b._compile_expr(gen.iter)
        iter_call = namer.fresh_temp("_iter")
        b._emit(CallNode(
            result=iter_call,
            func=SSAVar("iter"),
            args=[iter_var],
            source_loc=loc,
        ))

        loop_header = b._new_block("comp_loop")
        loop_body = b._new_block("comp_body")
        loop_end = b._new_block("comp_end")

        b._seal_and_jump(loop_header)
        b._current_block = loop_header
        b._ensure_block(loop_header)

        next_var = namer.fresh_temp("_next")
        has_next = namer.fresh_temp("_has")
        b._emit(CallNode(
            result=next_var,
            func=SSAVar("next"),
            args=[iter_call],
            source_loc=loc,
        ))
        b._emit(BranchNode(
            condition=has_next,
            true_block=loop_body,
            false_block=loop_end,
            source_loc=loc,
        ))
        b._cfg.add_edge(loop_header, loop_body)
        b._cfg.add_edge(loop_header, loop_end)

        b._current_block = loop_body
        b._ensure_block(loop_body)
        b._assign_target(gen.target, next_var)

        # if-conditions on the comprehension
        for if_clause in gen.ifs:
            cond_var = b._compile_expr(if_clause)
            cond_bool = b._truthiness.coerce(cond_var, b._current_bb, if_clause)
            filter_body = b._new_block("comp_filter")
            filter_skip = b._new_block("comp_skip")
            b._emit(BranchNode(
                condition=cond_bool,
                true_block=filter_body,
                false_block=filter_skip,
                source_loc=b._loc(if_clause),
            ))
            b._cfg.add_edge(b._current_block_label, filter_body)
            b._cfg.add_edge(b._current_block_label, filter_skip)

            b._current_block = filter_body
            b._ensure_block(filter_body)
            # skip block just jumps back to loop
            b._ensure_block(filter_skip)
            old_block = b._current_block_label
            b._current_block = filter_skip
            b._seal_and_jump(loop_header)
            b._current_block = old_block

        self._compile_generators_body(generators, idx + 1, body_fn, loc)

        b._seal_and_jump(loop_header)
        b._current_block = loop_end
        b._ensure_block(loop_end)

    # -- decorator wrapping --------------------------------------------------

    def wrap_with_decorators(
        self,
        func_var: SSAVar,
        decorators: List[ast.expr],
        loc: SourceLocation,
    ) -> SSAVar:
        b = self._builder
        current = func_var
        for dec in reversed(decorators):
            dec_var = b._compile_expr(dec)
            wrapped = b._namer.fresh_temp("_decorated")
            b._emit(CallNode(
                result=wrapped,
                func=dec_var,
                args=[current],
                source_loc=loc,
            ))
            current = wrapped
        return current

    # -- context manager (with) → try/finally -------------------------------

    def desugar_with(
        self,
        items: List[ast.withitem],
        body: List[ast.stmt],
        loc: SourceLocation,
        is_async: bool = False,
    ) -> None:
        b = self._builder
        namer = b._namer

        mgr_vars: List[SSAVar] = []
        enter_vars: List[SSAVar] = []

        for item in items:
            mgr = b._compile_expr(item.context_expr)
            mgr_vars.append(mgr)

            enter_attr = "__aenter__" if is_async else "__enter__"
            exit_attr = "__aexit__" if is_async else "__exit__"

            enter_fn = namer.fresh_temp("_enter_fn")
            b._emit(LoadAttrNode(
                dst=enter_fn, obj=mgr, attr=enter_attr, source_loc=loc,
            ))
            enter_result = namer.fresh_temp("_enter")
            b._emit(CallNode(
                result=enter_result, func=enter_fn, source_loc=loc,
            ))

            if is_async:
                awaited = namer.fresh_temp("_aenter")
                b._emit(AwaitNode(dst=awaited, value=enter_result, source_loc=loc))
                enter_result = awaited

            enter_vars.append(enter_result)

            if item.optional_vars:
                b._assign_target(item.optional_vars, enter_result)

        # try body
        try_block = b._new_block("with_body")
        finally_block = b._new_block("with_finally")
        merge_block = b._new_block("with_merge")

        b._seal_and_jump(try_block)
        b._current_block = try_block
        b._compile_body(body)

        b._seal_and_jump(finally_block)

        # finally: call __exit__
        b._current_block = finally_block
        b._ensure_block(finally_block)

        none_var = namer.fresh_temp("_none")
        b._emit(LiteralNode(
            dst=none_var, kind=LiteralKind.NONE, value=None, source_loc=loc,
        ))

        for mgr in reversed(mgr_vars):
            exit_attr = "__aexit__" if is_async else "__exit__"
            exit_fn = namer.fresh_temp("_exit_fn")
            b._emit(LoadAttrNode(
                dst=exit_fn, obj=mgr, attr=exit_attr, source_loc=loc,
            ))
            exit_result = namer.fresh_temp("_exit")
            b._emit(CallNode(
                result=exit_result,
                func=exit_fn,
                args=[none_var, none_var, none_var],
                source_loc=loc,
            ))
            if is_async:
                awaited = namer.fresh_temp("_aexit")
                b._emit(AwaitNode(dst=awaited, value=exit_result, source_loc=loc))

        b._seal_and_jump(merge_block)
        b._current_block = merge_block
        b._ensure_block(merge_block)

    # -- augmented assignment ------------------------------------------------

    def desugar_aug_assign(self, node: ast.AugAssign) -> None:
        b = self._builder
        loc = b._loc(node)
        right = b._compile_expr(node.value)

        op = _AUGASSIGN_OP_MAP.get(type(node.op), BinOp.ADD)

        if isinstance(node.target, ast.Name):
            left = b._namer.current(node.target.id)
            result = b._namer.fresh_temp("_aug")
            b._emit(BinOpNode(dst=result, op=op, left=left, right=right, source_loc=loc))
            new_var = b._namer.fresh_version(node.target.id)
            b._emit(AssignNode(dst=new_var, src=result, source_loc=loc))

        elif isinstance(node.target, ast.Attribute):
            obj = b._compile_expr(node.target.value)
            old_val = b._namer.fresh_temp("_load")
            b._emit(LoadAttrNode(
                dst=old_val, obj=obj, attr=node.target.attr, source_loc=loc,
            ))
            result = b._namer.fresh_temp("_aug")
            b._emit(BinOpNode(
                dst=result, op=op, left=old_val, right=right, source_loc=loc,
            ))
            b._emit(StoreAttrNode(
                obj=obj, attr=node.target.attr, value=result, source_loc=loc,
            ))

        elif isinstance(node.target, ast.Subscript):
            obj = b._compile_expr(node.target.value)
            idx = b._compile_expr(node.target.slice)
            old_val = b._namer.fresh_temp("_load")
            b._emit(IndexNode(
                dst=old_val, obj=obj, index=idx, source_loc=loc,
            ))
            result = b._namer.fresh_temp("_aug")
            b._emit(BinOpNode(
                dst=result, op=op, left=old_val, right=right, source_loc=loc,
            ))
            b._emit(StoreIndexNode(
                obj=obj, index=idx, value=result, source_loc=loc,
            ))

    # -- chained comparison → pairwise conjunction --------------------------

    def desugar_chained_compare(self, node: ast.Compare) -> SSAVar:
        b = self._builder
        loc = b._loc(node)
        namer = b._namer

        if len(node.ops) == 1:
            left = b._compile_expr(node.left)
            right = b._compile_expr(node.comparators[0])
            dst = namer.fresh_temp("_cmp")
            b._emit(CompareNode(
                dst=dst,
                ops=[_COMPARE_OP_MAP[type(node.ops[0])]],
                operands=[left, right],
                source_loc=loc,
            ))
            return dst

        # a < b < c  →  (a < b) and (b < c) with b evaluated once
        parts: List[SSAVar] = []
        left = b._compile_expr(node.left)

        for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            right = b._compile_expr(comparator)
            part_dst = namer.fresh_temp("_cmp")
            b._emit(CompareNode(
                dst=part_dst,
                ops=[_COMPARE_OP_MAP[type(op)]],
                operands=[left, right],
                source_loc=loc,
            ))
            parts.append(part_dst)
            left = right

        # combine with short-circuit AND
        result = parts[0]
        for part in parts[1:]:
            combined = namer.fresh_temp("_chain")
            b._emit(BinOpNode(
                dst=combined, op=BinOp.AND, left=result, right=part,
                source_loc=loc,
            ))
            result = combined
        return result

    # -- boolean short-circuit → conditional branches -----------------------

    def desugar_boolop(self, node: ast.BoolOp) -> SSAVar:
        b = self._builder
        namer = b._namer
        loc = b._loc(node)

        is_and = isinstance(node.op, ast.And)
        result_var = namer.fresh_temp("_boolop")

        values = node.values
        current_val = b._compile_expr(values[0])

        for i in range(1, len(values)):
            cond = b._truthiness.coerce(current_val, b._current_bb, node)
            true_block = b._new_block("sc_true")
            false_block = b._new_block("sc_false")
            merge_block = b._new_block("sc_merge")

            if is_and:
                b._emit(BranchNode(
                    condition=cond,
                    true_block=true_block,
                    false_block=false_block,
                    source_loc=loc,
                ))
            else:
                b._emit(BranchNode(
                    condition=cond,
                    true_block=false_block,
                    false_block=true_block,
                    source_loc=loc,
                ))

            pred_block = b._current_block_label
            b._cfg.add_edge(pred_block, true_block)
            b._cfg.add_edge(pred_block, false_block)

            # true_block: evaluate next value
            b._current_block = true_block
            b._ensure_block(true_block)
            next_val = b._compile_expr(values[i])
            true_end = b._current_block_label
            b._seal_and_jump(merge_block)

            # false_block: short-circuit with current value
            b._current_block = false_block
            b._ensure_block(false_block)
            false_end = b._current_block_label
            b._seal_and_jump(merge_block)

            # merge
            b._current_block = merge_block
            b._ensure_block(merge_block)

            phi_dst = namer.fresh_temp("_sc")
            b._emit(PhiNode(
                dst=phi_dst,
                incoming={true_end: next_val, false_end: current_val},
                source_loc=loc,
            ))
            current_val = phi_dst

        b._emit(AssignNode(dst=result_var, src=current_val, source_loc=loc))
        return result_var


# ---------------------------------------------------------------------------
# Function compiler
# ---------------------------------------------------------------------------

class FunctionCompiler:
    """Compiles a single function definition including parameter handling."""

    def __init__(self, parent: "SSABuilder", module: IRModule) -> None:
        self._parent = parent
        self._module = module

    def compile(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> IRFunction:
        loc = self._parent._loc(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        func = IRFunction(
            name=node.name,
            is_async=is_async,
            source_file=self._parent._source_file,
            source_line=getattr(node, "lineno", 0),
        )

        # parameters
        func.params = self._compile_params(node.args)

        # return annotation
        if node.returns:
            func.return_annotation = self._parent._annotation_str(node.returns)

        # decorators
        func.decorators = [
            ast.dump(d) if not isinstance(d, ast.Name) else d.id
            for d in node.decorator_list
        ]

        # detect generator
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                func.is_generator = True
                break

        # compile body with a fresh SSABuilder
        body_builder = SSABuilder(
            source_file=self._parent._source_file,
            module=self._module,
        )
        # set up parameter variables in the new scope
        for param in func.params:
            body_builder._namer.fresh_version(param.name)

        # capture closure variables
        scope = self._parent._scope_map.get(id(node))
        if scope:
            for free_name in scope.free_vars:
                outer = self._parent._namer.current(free_name)
                inner = body_builder._namer.fresh_version(free_name)
                body_builder._emit(ClosureCaptureNode(
                    dst=inner,
                    outer_var=outer,
                    is_nonlocal=free_name in scope.nonlocal_vars,
                    enclosing_scope=self._parent._current_function_name,
                    source_loc=loc,
                ))
                func.closure_captures.append(outer)

        body_builder._compile_body(node.body)

        # ensure there's a return at the end
        last_block = body_builder._current_bb
        if not last_block.terminator:
            none_var = body_builder._namer.fresh_temp("_none")
            body_builder._emit(LiteralNode(
                dst=none_var, kind=LiteralKind.NONE, value=None, source_loc=loc,
            ))
            body_builder._emit(ReturnNode(value=none_var, source_loc=loc))

        func.cfg = body_builder._cfg
        return func

    def _compile_params(self, args: ast.arguments) -> List[IRParameter]:
        params: List[IRParameter] = []

        all_args = args.posonlyargs + args.args

        # positional defaults are right-aligned
        n_defaults = len(args.defaults)
        n_positional = len(all_args)
        default_offset = n_positional - n_defaults

        for i, arg in enumerate(all_args):
            ann = self._parent._annotation_str(arg.annotation) if arg.annotation else None
            default_var = None
            if i >= default_offset:
                default_ast = args.defaults[i - default_offset]
                default_var = self._parent._compile_expr(default_ast)
            params.append(IRParameter(name=arg.arg, annotation=ann, default_var=default_var))

        if args.vararg:
            ann = self._parent._annotation_str(args.vararg.annotation) if args.vararg.annotation else None
            params.append(IRParameter(name=args.vararg.arg, annotation=ann, is_star=True))

        for i, arg in enumerate(args.kwonlyargs):
            ann = self._parent._annotation_str(arg.annotation) if arg.annotation else None
            default_var = None
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                default_var = self._parent._compile_expr(args.kw_defaults[i])
            params.append(IRParameter(name=arg.arg, annotation=ann, default_var=default_var))

        if args.kwarg:
            ann = self._parent._annotation_str(args.kwarg.annotation) if args.kwarg.annotation else None
            params.append(IRParameter(
                name=args.kwarg.arg, annotation=ann, is_double_star=True,
            ))

        return params


# ---------------------------------------------------------------------------
# Class compiler
# ---------------------------------------------------------------------------

class ClassCompiler:
    """Compiles a class body: methods, class variables, and bases."""

    def __init__(self, parent: "SSABuilder", module: IRModule) -> None:
        self._parent = parent
        self._module = module

    def compile(self, node: ast.ClassDef) -> IRClass:
        loc = self._parent._loc(node)

        bases = [self._parent._annotation_str(b) for b in node.bases]
        decorators = [
            ast.dump(d) if not isinstance(d, ast.Name) else d.id
            for d in node.decorator_list
        ]

        cls = IRClass(
            name=node.name,
            bases=bases,
            decorators=decorators,
            source_file=self._parent._source_file,
            source_line=getattr(node, "lineno", 0),
        )

        func_compiler = FunctionCompiler(self._parent, self._module)

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ir_fn = func_compiler.compile(stmt)
                cls.methods[stmt.name] = ir_fn
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        val = self._parent._compile_expr(stmt.value)
                        cls.class_vars[target.id] = val
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.target and isinstance(stmt.target, ast.Name):
                    if stmt.value:
                        val = self._parent._compile_expr(stmt.value)
                        cls.class_vars[stmt.target.id] = val

        return cls


# ---------------------------------------------------------------------------
# AST operator mapping tables
# ---------------------------------------------------------------------------

_BINOP_MAP: Dict[type, BinOp] = {
    ast.Add: BinOp.ADD,
    ast.Sub: BinOp.SUB,
    ast.Mult: BinOp.MUL,
    ast.Div: BinOp.DIV,
    ast.FloorDiv: BinOp.FLOOR_DIV,
    ast.Mod: BinOp.MOD,
    ast.Pow: BinOp.POW,
    ast.LShift: BinOp.LSHIFT,
    ast.RShift: BinOp.RSHIFT,
    ast.BitAnd: BinOp.BIT_AND,
    ast.BitOr: BinOp.BIT_OR,
    ast.BitXor: BinOp.BIT_XOR,
    ast.MatMult: BinOp.MAT_MUL,
}

_AUGASSIGN_OP_MAP: Dict[type, BinOp] = _BINOP_MAP.copy()

_UNARYOP_MAP: Dict[type, UnaryOp] = {
    ast.Not: UnaryOp.NOT,
    ast.USub: UnaryOp.NEGATE,
    ast.UAdd: UnaryOp.POS,
    ast.Invert: UnaryOp.INVERT,
}

_COMPARE_OP_MAP: Dict[type, CompareOp] = {
    ast.Lt: CompareOp.LT,
    ast.LtE: CompareOp.LE,
    ast.Gt: CompareOp.GT,
    ast.GtE: CompareOp.GE,
    ast.Eq: CompareOp.EQ,
    ast.NotEq: CompareOp.NE,
    ast.Is: CompareOp.IS,
    ast.IsNot: CompareOp.IS_NOT,
    ast.In: CompareOp.IN,
    ast.NotIn: CompareOp.NOT_IN,
}

_LITERAL_KIND_MAP: Dict[type, LiteralKind] = {
    int: LiteralKind.INT,
    float: LiteralKind.FLOAT,
    str: LiteralKind.STR,
    bytes: LiteralKind.BYTES,
    bool: LiteralKind.BOOL,
    complex: LiteralKind.COMPLEX,
}


# ---------------------------------------------------------------------------
# SSABuilder — core AST → SSA IR translation
# ---------------------------------------------------------------------------

class SSABuilder:
    """Translates a Python AST into SSA-form IR with basic blocks,
    phi nodes at join points, and explicit truthiness coercion."""

    def __init__(
        self,
        source_file: str = "<unknown>",
        module: Optional[IRModule] = None,
    ) -> None:
        self._source_file = source_file
        self._module = module or IRModule(source_file=source_file)
        self._namer = _SSANamer()
        self._cfg = CFG()
        self._block_counter = 0
        self._current_function_name: Optional[str] = None
        self._scope_map: Dict[int, _Scope] = {}

        # current block
        entry = self._new_block("entry")
        self._current_block: str = entry
        self._ensure_block(entry)
        self._cfg.entry = entry

        # helpers
        self._truthiness = TruthinessModeler(self._namer, self._loc)
        self._desugarer = Desugarer(self)
        self._exc_handler = ExceptionFlowHandler(self)
        self._async_desugarer = AsyncDesugarer(self)

        # loop context for break/continue
        self._loop_stack: List[Tuple[str, str]] = []  # (header, exit)
        self._break_targets: List[str] = []
        self._continue_targets: List[str] = []

    # -- block management ----------------------------------------------------

    def _new_block(self, prefix: str = "bb") -> str:
        self._block_counter += 1
        return f"{prefix}_{self._block_counter}"

    def _ensure_block(self, label: str) -> IRBasicBlock:
        if label not in self._cfg.blocks:
            block = IRBasicBlock(label=label)
            self._cfg.add_block(block)
        return self._cfg.blocks[label]

    @property
    def _current_bb(self) -> IRBasicBlock:
        return self._ensure_block(self._current_block)

    @property
    def _current_block_label(self) -> str:
        return self._current_block

    def _emit(self, node: IRNode) -> None:
        self._current_bb.append(node)

    def _seal_and_jump(self, target: str) -> None:
        bb = self._current_bb
        if not bb.terminator:
            bb.append(JumpNode(target=target))
            self._cfg.add_edge(bb.label, target)
            self._ensure_block(target)

    # -- source location -----------------------------------------------------

    def _loc(self, node: Optional[ast.AST] = None) -> SourceLocation:
        if node is None:
            return SourceLocation(file=self._source_file)
        return SourceLocation(
            file=self._source_file,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", 0) or 0,
            end_col=getattr(node, "end_col_offset", 0) or 0,
        )

    # -- annotation extraction -----------------------------------------------

    def _annotation_str(self, node: Optional[ast.AST]) -> str:
        if node is None:
            return "unknown"
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._annotation_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            val = self._annotation_str(node.value)
            sl = self._annotation_str(node.slice)
            return f"{val}[{sl}]"
        if isinstance(node, ast.Tuple):
            parts = ", ".join(self._annotation_str(e) for e in node.elts)
            return parts
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._annotation_str(node.left)
            right = self._annotation_str(node.right)
            return f"{left} | {right}"
        return ast.dump(node)

    # -- compile body (statement list) ---------------------------------------

    def _compile_body(self, stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            self._compile_stmt(stmt)

    # -- statement compilation -----------------------------------------------

    def _compile_stmt(self, node: ast.stmt) -> None:
        loc = self._loc(node)

        if isinstance(node, ast.Assign):
            self._compile_assign(node, loc)
        elif isinstance(node, ast.AnnAssign):
            self._compile_ann_assign(node, loc)
        elif isinstance(node, ast.AugAssign):
            self._desugarer.desugar_aug_assign(node)
        elif isinstance(node, ast.Return):
            self._compile_return(node, loc)
        elif isinstance(node, ast.If):
            self._compile_if(node, loc)
        elif isinstance(node, ast.While):
            self._compile_while(node, loc)
        elif isinstance(node, ast.For):
            self._compile_for(node, loc)
        elif isinstance(node, ast.AsyncFor):
            self._async_desugarer.compile_async_for(node)
        elif isinstance(node, ast.With):
            self._compile_with_common(node)
        elif isinstance(node, ast.AsyncWith):
            self._async_desugarer.compile_async_with(node)
        elif isinstance(node, ast.Try):
            self._exc_handler.compile_try(node, loc)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._compile_funcdef(node, loc)
        elif isinstance(node, ast.ClassDef):
            self._compile_classdef(node, loc)
        elif isinstance(node, ast.Expr):
            self._compile_expr(node.value)
        elif isinstance(node, ast.Delete):
            self._compile_delete(node, loc)
        elif isinstance(node, ast.Raise):
            self._compile_raise(node, loc)
        elif isinstance(node, ast.Assert):
            self._compile_assert(node, loc)
        elif isinstance(node, ast.Import):
            self._compile_import(node, loc)
        elif isinstance(node, ast.ImportFrom):
            self._compile_import_from(node, loc)
        elif isinstance(node, ast.Global):
            pass  # handled by ScopeResolver
        elif isinstance(node, ast.Nonlocal):
            pass  # handled by ScopeResolver
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            self._compile_break(loc)
        elif isinstance(node, ast.Continue):
            self._compile_continue(loc)
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):  # 3.10+
            self._compile_match(node, loc)
        elif hasattr(ast, "TryStar") and isinstance(node, getattr(ast, "TryStar")):
            # Python 3.11+ except* — treat like regular try for now
            self._exc_handler.compile_try(node, loc)

    # -- assignments ---------------------------------------------------------

    def _compile_assign(self, node: ast.Assign, loc: SourceLocation) -> None:
        value = self._compile_expr(node.value)
        for target in node.targets:
            self._assign_target(target, value)

    def _compile_ann_assign(self, node: ast.AnnAssign, loc: SourceLocation) -> None:
        if node.value:
            value = self._compile_expr(node.value)
            if node.target:
                self._assign_target(node.target, value)

    def _assign_target(self, target: ast.AST, value: SSAVar) -> None:
        loc = self._loc(target)
        if isinstance(target, ast.Name):
            dst = self._namer.fresh_version(target.id)
            self._emit(AssignNode(dst=dst, src=value, source_loc=loc))
        elif isinstance(target, ast.Attribute):
            obj = self._compile_expr(target.value)
            self._emit(StoreAttrNode(
                obj=obj, attr=target.attr, value=value, source_loc=loc,
            ))
        elif isinstance(target, ast.Subscript):
            obj = self._compile_expr(target.value)
            idx = self._compile_expr(target.slice)
            self._emit(StoreIndexNode(
                obj=obj, index=idx, value=value, source_loc=loc,
            ))
        elif isinstance(target, (ast.Tuple, ast.List)):
            star_idx: Optional[int] = None
            target_vars: List[SSAVar] = []
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Starred):
                    star_idx = i
                    var = self._namer.fresh_version(
                        elt.value.id if isinstance(elt.value, ast.Name) else f"_star{i}"
                    )
                else:
                    var = self._namer.fresh_temp("_unpack")
                target_vars.append(var)

            self._emit(UnpackNode(
                targets=target_vars,
                src=value,
                star_index=star_idx,
                source_loc=loc,
            ))

            # assign individual elements to their names
            for i, elt in enumerate(target.elts):
                actual_elt = elt.value if isinstance(elt, ast.Starred) else elt
                if isinstance(actual_elt, ast.Name):
                    if not isinstance(elt, ast.Starred):
                        dst = self._namer.fresh_version(actual_elt.id)
                        self._emit(AssignNode(
                            dst=dst, src=target_vars[i], source_loc=loc,
                        ))
                elif isinstance(actual_elt, (ast.Tuple, ast.List)):
                    self._assign_target(actual_elt, target_vars[i])

    # -- return --------------------------------------------------------------

    def _compile_return(self, node: ast.Return, loc: SourceLocation) -> None:
        value = None
        if node.value:
            value = self._compile_expr(node.value)
        self._emit(ReturnNode(value=value, source_loc=loc))

    # -- if/elif/else --------------------------------------------------------

    def _compile_if(self, node: ast.If, loc: SourceLocation) -> None:
        cond = self._compile_expr(node.test)
        cond_bool = self._truthiness.coerce(cond, self._current_bb, node.test)

        true_block = self._new_block("if_true")
        false_block = self._new_block("if_false")
        merge_block = self._new_block("if_merge")

        self._emit(BranchNode(
            condition=cond_bool,
            true_block=true_block,
            false_block=false_block,
            source_loc=loc,
        ))
        pred = self._current_block_label
        self._cfg.add_edge(pred, true_block)
        self._cfg.add_edge(pred, false_block)

        snap = self._namer.snapshot()

        # true branch
        self._current_block = true_block
        self._ensure_block(true_block)
        self._compile_body(node.body)
        true_end = self._current_block_label
        self._seal_and_jump(merge_block)
        snap_true = self._namer.snapshot()

        # false branch
        self._namer.restore_snapshot(snap)
        self._current_block = false_block
        self._ensure_block(false_block)
        if node.orelse:
            self._compile_body(node.orelse)
        false_end = self._current_block_label
        self._seal_and_jump(merge_block)
        snap_false = self._namer.snapshot()

        # merge with phi nodes
        self._current_block = merge_block
        self._ensure_block(merge_block)
        self._insert_phis_for_branches(
            snap, snap_true, snap_false, true_end, false_end, loc,
        )

    def _insert_phis_for_branches(
        self,
        snap_before: Dict[str, int],
        snap_true: Dict[str, int],
        snap_false: Dict[str, int],
        true_label: str,
        false_label: str,
        loc: SourceLocation,
    ) -> None:
        modified = set(snap_true.keys()) | set(snap_false.keys())
        for name in modified:
            if name.startswith("_t") or name.startswith("_bool") or name.startswith("_"):
                continue
            tv = snap_true.get(name)
            fv = snap_false.get(name)
            if tv is not None and fv is not None and tv != fv:
                incoming = {
                    true_label: SSAVar(name, tv),
                    false_label: SSAVar(name, fv),
                }
                phi_dst = self._namer.fresh_version(name)
                self._emit(PhiNode(dst=phi_dst, incoming=incoming, source_loc=loc))

    # -- while ---------------------------------------------------------------

    def _compile_while(self, node: ast.While, loc: SourceLocation) -> None:
        header = self._new_block("while_header")
        body_block = self._new_block("while_body")
        else_block = self._new_block("while_else") if node.orelse else None
        exit_block = self._new_block("while_exit")

        self._seal_and_jump(header)

        # header: evaluate condition
        self._current_block = header
        self._ensure_block(header)
        cond = self._compile_expr(node.test)
        cond_bool = self._truthiness.coerce(cond, self._current_bb, node.test)

        false_target = else_block if else_block else exit_block
        self._emit(BranchNode(
            condition=cond_bool,
            true_block=body_block,
            false_block=false_target,
            source_loc=loc,
        ))
        self._cfg.add_edge(header, body_block)
        self._cfg.add_edge(header, false_target)

        # body
        self._loop_stack.append((header, exit_block))
        self._current_block = body_block
        self._ensure_block(body_block)
        self._compile_body(node.body)
        self._seal_and_jump(header)
        self._loop_stack.pop()

        # else
        if else_block:
            self._current_block = else_block
            self._ensure_block(else_block)
            self._compile_body(node.orelse)
            self._seal_and_jump(exit_block)

        self._current_block = exit_block
        self._ensure_block(exit_block)

    # -- for -----------------------------------------------------------------

    def _compile_for(self, node: ast.For, loc: SourceLocation) -> None:
        iter_expr = self._compile_expr(node.iter)
        iter_var = self._namer.fresh_temp("_iter")
        self._emit(CallNode(
            result=iter_var,
            func=SSAVar("iter"),
            args=[iter_expr],
            source_loc=loc,
        ))
        self._compile_for_common(node, iter_var)

    def _compile_for_common(
        self,
        node: Union[ast.For, ast.AsyncFor],
        iter_var: SSAVar,
        is_async: bool = False,
    ) -> None:
        loc = self._loc(node)
        header = self._new_block("for_header")
        body_block = self._new_block("for_body")
        else_block = self._new_block("for_else") if node.orelse else None
        exit_block = self._new_block("for_exit")

        self._seal_and_jump(header)
        self._current_block = header
        self._ensure_block(header)

        # __next__ call
        next_result = self._namer.fresh_temp("_next")
        has_next = self._namer.fresh_temp("_has_next")
        self._emit(CallNode(
            result=next_result,
            func=SSAVar("next"),
            args=[iter_var],
            source_loc=loc,
        ))

        false_target = else_block if else_block else exit_block
        self._emit(BranchNode(
            condition=has_next,
            true_block=body_block,
            false_block=false_target,
            source_loc=loc,
        ))
        self._cfg.add_edge(header, body_block)
        self._cfg.add_edge(header, false_target)

        # body
        self._loop_stack.append((header, exit_block))
        self._current_block = body_block
        self._ensure_block(body_block)

        if is_async:
            awaited = self._namer.fresh_temp("_anext")
            self._emit(AwaitNode(dst=awaited, value=next_result, source_loc=loc))
            self._assign_target(node.target, awaited)
        else:
            self._assign_target(node.target, next_result)

        self._compile_body(node.body)
        self._seal_and_jump(header)
        self._loop_stack.pop()

        # else
        if else_block:
            self._current_block = else_block
            self._ensure_block(else_block)
            self._compile_body(node.orelse)
            self._seal_and_jump(exit_block)

        self._current_block = exit_block
        self._ensure_block(exit_block)

    # -- with ----------------------------------------------------------------

    def _compile_with_common(
        self,
        node: Union[ast.With, ast.AsyncWith],
        is_async: bool = False,
    ) -> None:
        self._desugarer.desugar_with(
            node.items, node.body, self._loc(node), is_async=is_async,
        )

    # -- break / continue ----------------------------------------------------

    def _compile_break(self, loc: SourceLocation) -> None:
        if self._loop_stack:
            _, exit_block = self._loop_stack[-1]
            self._seal_and_jump(exit_block)
            dead = self._new_block("after_break")
            self._current_block = dead
            self._ensure_block(dead)

    def _compile_continue(self, loc: SourceLocation) -> None:
        if self._loop_stack:
            header, _ = self._loop_stack[-1]
            self._seal_and_jump(header)
            dead = self._new_block("after_continue")
            self._current_block = dead
            self._ensure_block(dead)

    # -- function/class defs -------------------------------------------------

    def _compile_funcdef(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        loc: SourceLocation,
    ) -> None:
        func_compiler = FunctionCompiler(self, self._module)
        ir_func = func_compiler.compile(node)
        self._module.functions[node.name] = ir_func

        # bind function name in current scope
        func_var = self._namer.fresh_version(node.name)
        self._emit(LiteralNode(
            dst=func_var,
            kind=LiteralKind.STR,
            value=f"<function {node.name}>",
            source_loc=loc,
        ))

        # wrap with decorators
        if node.decorator_list:
            wrapped = self._desugarer.wrap_with_decorators(
                func_var, node.decorator_list, loc,
            )
            final = self._namer.fresh_version(node.name)
            self._emit(AssignNode(dst=final, src=wrapped, source_loc=loc))

    def _compile_classdef(self, node: ast.ClassDef, loc: SourceLocation) -> None:
        cls_compiler = ClassCompiler(self, self._module)
        ir_cls = cls_compiler.compile(node)
        self._module.classes[node.name] = ir_cls

        class_var = self._namer.fresh_version(node.name)
        self._emit(LiteralNode(
            dst=class_var,
            kind=LiteralKind.STR,
            value=f"<class {node.name}>",
            source_loc=loc,
        ))

        if node.decorator_list:
            wrapped = self._desugarer.wrap_with_decorators(
                class_var, node.decorator_list, loc,
            )
            final = self._namer.fresh_version(node.name)
            self._emit(AssignNode(dst=final, src=wrapped, source_loc=loc))

    # -- delete / raise / assert / import ------------------------------------

    def _compile_delete(self, node: ast.Delete, loc: SourceLocation) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                var = self._namer.current(target.id)
                self._emit(DeleteNode(target=var, source_loc=loc))
            elif isinstance(target, ast.Attribute):
                obj = self._compile_expr(target.value)
                self._emit(DeleteNode(
                    target=obj, attr=target.attr, source_loc=loc,
                ))
            elif isinstance(target, ast.Subscript):
                obj = self._compile_expr(target.value)
                idx = self._compile_expr(target.slice)
                self._emit(DeleteNode(target=obj, source_loc=loc))

    def _compile_raise(self, node: ast.Raise, loc: SourceLocation) -> None:
        exc = self._compile_expr(node.exc) if node.exc else None
        cause = self._compile_expr(node.cause) if node.cause else None
        self._emit(RaiseNode(exc=exc, cause=cause, source_loc=loc))

    def _compile_assert(self, node: ast.Assert, loc: SourceLocation) -> None:
        test = self._compile_expr(node.test)
        msg = self._compile_expr(node.msg) if node.msg else None
        self._emit(AssertNode(test=test, msg=msg, source_loc=loc))

    def _compile_import(self, node: ast.Import, loc: SourceLocation) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            dst = self._namer.fresh_version(name)
            imp = ImportNode(
                dst=dst,
                module_path=alias.name,
                names=[(alias.name, alias.asname)],
                is_from=False,
                source_loc=loc,
            )
            self._emit(imp)
            self._module.imports.append(imp)

    def _compile_import_from(self, node: ast.ImportFrom, loc: SourceLocation) -> None:
        module_path = node.module or ""
        names: List[Tuple[str, Optional[str]]] = [
            (a.name, a.asname) for a in node.names
        ]
        for alias in node.names:
            local_name = alias.asname or alias.name
            dst = self._namer.fresh_version(local_name)
            imp = ImportNode(
                dst=dst,
                module_path=module_path,
                names=names,
                is_from=True,
                level=node.level or 0,
                source_loc=loc,
            )
            self._emit(imp)
            self._module.imports.append(imp)

    # -- match (Python 3.10+) ------------------------------------------------

    def _compile_match(self, node: Any, loc: SourceLocation) -> None:
        subject = self._compile_expr(node.subject)
        merge_block = self._new_block("match_merge")
        snap = self._namer.snapshot()
        case_snapshots: List[Tuple[str, Dict[str, int]]] = []

        for i, case in enumerate(node.cases):
            case_block = self._new_block(f"match_case_{i}")
            next_block = self._new_block(f"match_next_{i}")

            pattern = case.pattern
            guard_var = self._compile_match_pattern(pattern, subject, loc)

            if guard_var is not None:
                self._emit(BranchNode(
                    condition=guard_var,
                    true_block=case_block,
                    false_block=next_block,
                    source_loc=loc,
                ))
                self._cfg.add_edge(self._current_block_label, case_block)
                self._cfg.add_edge(self._current_block_label, next_block)
            else:
                self._seal_and_jump(case_block)

            # case body
            self._current_block = case_block
            self._ensure_block(case_block)

            if case.guard:
                guard_cond = self._compile_expr(case.guard)
                guard_bool = self._truthiness.coerce(
                    guard_cond, self._current_bb, case.guard,
                )
                guarded = self._new_block(f"match_guarded_{i}")
                self._emit(BranchNode(
                    condition=guard_bool,
                    true_block=guarded,
                    false_block=next_block,
                    source_loc=loc,
                ))
                self._cfg.add_edge(case_block, guarded)
                self._cfg.add_edge(case_block, next_block)
                self._current_block = guarded
                self._ensure_block(guarded)

            self._compile_body(case.body)
            case_end = self._current_block_label
            case_snapshots.append((case_end, self._namer.snapshot()))
            self._seal_and_jump(merge_block)

            self._namer.restore_snapshot(snap)
            self._current_block = next_block
            self._ensure_block(next_block)

        self._seal_and_jump(merge_block)
        self._current_block = merge_block
        self._ensure_block(merge_block)

    def _compile_match_pattern(
        self, pattern: Any, subject: SSAVar, loc: SourceLocation,
    ) -> Optional[SSAVar]:
        """Compile a match pattern, returning a guard SSAVar or None for wildcard."""
        pattern_type = type(pattern).__name__

        if pattern_type == "MatchValue":
            val = self._compile_expr(pattern.value)
            dst = self._namer.fresh_temp("_match")
            self._emit(CompareNode(
                dst=dst,
                ops=[CompareOp.EQ],
                operands=[subject, val],
                source_loc=loc,
            ))
            return dst

        elif pattern_type == "MatchSingleton":
            val = self._namer.fresh_temp("_sing")
            if pattern.value is None:
                self._emit(LiteralNode(
                    dst=val, kind=LiteralKind.NONE, value=None, source_loc=loc,
                ))
            elif pattern.value is True:
                self._emit(LiteralNode(
                    dst=val, kind=LiteralKind.BOOL, value=True, source_loc=loc,
                ))
            else:
                self._emit(LiteralNode(
                    dst=val, kind=LiteralKind.BOOL, value=False, source_loc=loc,
                ))
            dst = self._namer.fresh_temp("_match")
            self._emit(CompareNode(
                dst=dst,
                ops=[CompareOp.IS],
                operands=[subject, val],
                source_loc=loc,
            ))
            return dst

        elif pattern_type == "MatchClass":
            cls_var = self._compile_expr(pattern.cls)
            dst = self._namer.fresh_temp("_isinstance")
            self._emit(TypeTestNode(
                dst=dst,
                subject=subject,
                tested_type=self._annotation_str(pattern.cls),
                source_loc=loc,
            ))
            return dst

        elif pattern_type == "MatchAs":
            if pattern.name:
                v = self._namer.fresh_version(pattern.name)
                self._emit(AssignNode(dst=v, src=subject, source_loc=loc))
            if pattern.pattern:
                return self._compile_match_pattern(pattern.pattern, subject, loc)
            return None  # wildcard

        elif pattern_type == "MatchOr":
            parts: List[SSAVar] = []
            for p in pattern.patterns:
                pv = self._compile_match_pattern(p, subject, loc)
                if pv is not None:
                    parts.append(pv)
            if not parts:
                return None
            result = parts[0]
            for p in parts[1:]:
                combined = self._namer.fresh_temp("_or")
                self._emit(BinOpNode(
                    dst=combined, op=BinOp.OR, left=result, right=p,
                    source_loc=loc,
                ))
                result = combined
            return result

        return None  # MatchStar, MatchMapping, MatchSequence — wildcard fallback

    # ======================================================================
    # Expression compilation
    # ======================================================================

    def _compile_expr(self, node: ast.expr) -> SSAVar:
        loc = self._loc(node)

        if isinstance(node, ast.Constant):
            return self._compile_constant(node, loc)

        elif isinstance(node, ast.Name):
            return self._namer.current(node.id)

        elif isinstance(node, ast.BinOp):
            return self._compile_binop(node, loc)

        elif isinstance(node, ast.UnaryOp):
            return self._compile_unaryop(node, loc)

        elif isinstance(node, ast.BoolOp):
            return self._desugarer.desugar_boolop(node)

        elif isinstance(node, ast.Compare):
            return self._desugarer.desugar_chained_compare(node)

        elif isinstance(node, ast.Call):
            return self._compile_call(node, loc)

        elif isinstance(node, ast.Attribute):
            return self._compile_attribute(node, loc)

        elif isinstance(node, ast.Subscript):
            return self._compile_subscript(node, loc)

        elif isinstance(node, ast.IfExp):
            return self._compile_ifexp(node, loc)

        elif isinstance(node, ast.Lambda):
            return self._compile_lambda(node, loc)

        elif isinstance(node, ast.Dict):
            return self._compile_dict(node, loc)

        elif isinstance(node, ast.List):
            return self._compile_list(node, loc)

        elif isinstance(node, ast.Tuple):
            return self._compile_tuple(node, loc)

        elif isinstance(node, ast.Set):
            return self._compile_set(node, loc)

        elif isinstance(node, ast.ListComp):
            return self._desugarer.desugar_listcomp(node)

        elif isinstance(node, ast.SetComp):
            return self._desugarer.desugar_setcomp(node)

        elif isinstance(node, ast.DictComp):
            return self._desugarer.desugar_dictcomp(node)

        elif isinstance(node, ast.GeneratorExp):
            return self._desugarer.desugar_genexp(node)

        elif isinstance(node, ast.Yield):
            return self._compile_yield(node, loc)

        elif isinstance(node, ast.YieldFrom):
            return self._compile_yield_from(node, loc)

        elif isinstance(node, ast.Await):
            return self._async_desugarer.compile_await(node)

        elif isinstance(node, ast.JoinedStr):
            return self._compile_fstring(node, loc)

        elif isinstance(node, ast.FormattedValue):
            return self._compile_expr(node.value)

        elif isinstance(node, ast.Starred):
            return self._compile_expr(node.value)

        elif isinstance(node, ast.NamedExpr):
            return self._compile_named_expr(node, loc)

        elif isinstance(node, ast.Slice):
            return self._compile_slice_expr(node, loc)

        # fallback: unknown expression
        return self._namer.fresh_temp("_unknown")

    # -- constants -----------------------------------------------------------

    def _compile_constant(self, node: ast.Constant, loc: SourceLocation) -> SSAVar:
        val = node.value
        if val is None:
            kind = LiteralKind.NONE
        elif val is ...:
            kind = LiteralKind.ELLIPSIS
        elif isinstance(val, bool):
            kind = LiteralKind.BOOL
        elif isinstance(val, int):
            kind = LiteralKind.INT
        elif isinstance(val, float):
            kind = LiteralKind.FLOAT
        elif isinstance(val, complex):
            kind = LiteralKind.COMPLEX
        elif isinstance(val, str):
            kind = LiteralKind.STR
        elif isinstance(val, bytes):
            kind = LiteralKind.BYTES
        else:
            kind = LiteralKind.STR
            val = repr(val)

        dst = self._namer.fresh_temp("_lit")
        self._emit(LiteralNode(dst=dst, kind=kind, value=val, source_loc=loc))
        return dst

    # -- binary / unary ops --------------------------------------------------

    def _compile_binop(self, node: ast.BinOp, loc: SourceLocation) -> SSAVar:
        left = self._compile_expr(node.left)
        right = self._compile_expr(node.right)
        op = _BINOP_MAP.get(type(node.op), BinOp.ADD)
        dst = self._namer.fresh_temp("_binop")
        self._emit(BinOpNode(dst=dst, op=op, left=left, right=right, source_loc=loc))
        return dst

    def _compile_unaryop(self, node: ast.UnaryOp, loc: SourceLocation) -> SSAVar:
        operand = self._compile_expr(node.operand)
        op = _UNARYOP_MAP.get(type(node.op), UnaryOp.NOT)
        dst = self._namer.fresh_temp("_unary")
        self._emit(UnaryOpNode(dst=dst, op=op, operand=operand, source_loc=loc))
        return dst

    # -- calls ---------------------------------------------------------------

    def _compile_call(self, node: ast.Call, loc: SourceLocation) -> SSAVar:
        # check for isinstance/hasattr/len specializations
        if isinstance(node.func, ast.Name):
            special = self._try_special_call(node, loc)
            if special is not None:
                return special

        # method call detection
        receiver: Optional[SSAVar] = None
        is_method = False
        if isinstance(node.func, ast.Attribute):
            receiver = self._compile_expr(node.func.value)
            func_var = self._namer.fresh_temp("_method")
            self._emit(LoadAttrNode(
                dst=func_var, obj=receiver, attr=node.func.attr, source_loc=loc,
            ))
            is_method = True
        else:
            func_var = self._compile_expr(node.func)

        args: List[SSAVar] = []
        kwargs: Dict[str, SSAVar] = {}
        star_args: Optional[SSAVar] = None
        star_kwargs: Optional[SSAVar] = None

        for arg in node.args:
            if isinstance(arg, ast.Starred):
                star_args = self._compile_expr(arg.value)
            else:
                args.append(self._compile_expr(arg))

        for kw in node.keywords:
            val = self._compile_expr(kw.value)
            if kw.arg is None:
                star_kwargs = val
            else:
                kwargs[kw.arg] = val

        result = self._namer.fresh_temp("_call")
        self._emit(CallNode(
            result=result,
            func=func_var,
            receiver=receiver,
            args=args,
            kwargs=kwargs,
            is_method=is_method,
            star_args=star_args,
            star_kwargs=star_kwargs,
            source_loc=loc,
        ))
        return result

    def _try_special_call(
        self, node: ast.Call, loc: SourceLocation,
    ) -> Optional[SSAVar]:
        """Handle isinstance, hasattr, len, type as specialized nodes."""
        assert isinstance(node.func, ast.Name)
        name = node.func.id

        if name == "isinstance" and len(node.args) == 2:
            subject = self._compile_expr(node.args[0])
            type_str = self._annotation_str(node.args[1])
            dst = self._namer.fresh_temp("_isinstance")
            self._emit(TypeTestNode(
                dst=dst, subject=subject, tested_type=type_str, source_loc=loc,
            ))
            return dst

        if name == "hasattr" and len(node.args) == 2:
            obj = self._compile_expr(node.args[0])
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                attr_name = node.args[1].value
            else:
                attr_name = "<dynamic>"
            dst = self._namer.fresh_temp("_hasattr")
            self._emit(HasAttrNode(
                dst=dst, obj=obj, attr_name=attr_name, source_loc=loc,
            ))
            return dst

        if name == "len" and len(node.args) == 1:
            obj = self._compile_expr(node.args[0])
            dst = self._namer.fresh_temp("_len")
            self._emit(LenNode(dst=dst, obj=obj, source_loc=loc))
            return dst

        return None

    # -- attribute / subscript -----------------------------------------------

    def _compile_attribute(self, node: ast.Attribute, loc: SourceLocation) -> SSAVar:
        obj = self._compile_expr(node.value)
        dst = self._namer.fresh_temp("_attr")
        self._emit(LoadAttrNode(dst=dst, obj=obj, attr=node.attr, source_loc=loc))
        return dst

    def _compile_subscript(self, node: ast.Subscript, loc: SourceLocation) -> SSAVar:
        obj = self._compile_expr(node.value)

        if isinstance(node.slice, ast.Slice):
            lower = self._compile_expr(node.slice.lower) if node.slice.lower else None
            upper = self._compile_expr(node.slice.upper) if node.slice.upper else None
            step = self._compile_expr(node.slice.step) if node.slice.step else None
            dst = self._namer.fresh_temp("_slice")
            self._emit(SliceNode(
                dst=dst, obj=obj, lower=lower, upper=upper, step=step,
                source_loc=loc,
            ))
            return dst

        idx = self._compile_expr(node.slice)
        dst = self._namer.fresh_temp("_idx")
        self._emit(IndexNode(dst=dst, obj=obj, index=idx, source_loc=loc))
        return dst

    # -- ternary / if expression ---------------------------------------------

    def _compile_ifexp(self, node: ast.IfExp, loc: SourceLocation) -> SSAVar:
        cond = self._compile_expr(node.test)
        cond_bool = self._truthiness.coerce(cond, self._current_bb, node.test)

        true_block = self._new_block("ternary_true")
        false_block = self._new_block("ternary_false")
        merge_block = self._new_block("ternary_merge")

        self._emit(BranchNode(
            condition=cond_bool,
            true_block=true_block,
            false_block=false_block,
            source_loc=loc,
        ))
        pred = self._current_block_label
        self._cfg.add_edge(pred, true_block)
        self._cfg.add_edge(pred, false_block)

        # true
        self._current_block = true_block
        self._ensure_block(true_block)
        true_val = self._compile_expr(node.body)
        true_end = self._current_block_label
        self._seal_and_jump(merge_block)

        # false
        self._current_block = false_block
        self._ensure_block(false_block)
        false_val = self._compile_expr(node.orelse)
        false_end = self._current_block_label
        self._seal_and_jump(merge_block)

        # merge
        self._current_block = merge_block
        self._ensure_block(merge_block)
        dst = self._namer.fresh_temp("_ternary")
        self._emit(PhiNode(
            dst=dst,
            incoming={true_end: true_val, false_end: false_val},
            source_loc=loc,
        ))
        return dst

    # -- lambda --------------------------------------------------------------

    def _compile_lambda(self, node: ast.Lambda, loc: SourceLocation) -> SSAVar:
        lam_name = f"<lambda>_{self._namer._temp_counter}"
        self._namer._temp_counter += 1

        # create a synthetic function def
        func = IRFunction(
            name=lam_name,
            is_async=False,
            source_file=self._source_file,
            source_line=getattr(node, "lineno", 0),
        )

        func_compiler = FunctionCompiler(self, self._module)
        func.params = func_compiler._compile_params(node.args)

        body_builder = SSABuilder(
            source_file=self._source_file,
            module=self._module,
        )
        for param in func.params:
            body_builder._namer.fresh_version(param.name)

        ret_val = body_builder._compile_expr(node.body)
        body_builder._emit(ReturnNode(value=ret_val, source_loc=loc))
        func.cfg = body_builder._cfg

        self._module.functions[lam_name] = func

        dst = self._namer.fresh_temp("_lambda")
        self._emit(LiteralNode(
            dst=dst, kind=LiteralKind.STR, value=lam_name, source_loc=loc,
        ))
        return dst

    # -- containers ----------------------------------------------------------

    def _compile_dict(self, node: ast.Dict, loc: SourceLocation) -> SSAVar:
        keys: List[Optional[SSAVar]] = []
        values: List[SSAVar] = []
        for k, v in zip(node.keys, node.values):
            keys.append(self._compile_expr(k) if k is not None else None)
            values.append(self._compile_expr(v))
        dst = self._namer.fresh_temp("_dict")
        self._emit(ContainerCreateNode(
            dst=dst, kind=ContainerKind.DICT, elements=values, keys=keys,
            source_loc=loc,
        ))
        return dst

    def _compile_list(self, node: ast.List, loc: SourceLocation) -> SSAVar:
        elts = [self._compile_expr(e) for e in node.elts]
        dst = self._namer.fresh_temp("_list")
        self._emit(ContainerCreateNode(
            dst=dst, kind=ContainerKind.LIST, elements=elts, source_loc=loc,
        ))
        return dst

    def _compile_tuple(self, node: ast.Tuple, loc: SourceLocation) -> SSAVar:
        elts = [self._compile_expr(e) for e in node.elts]
        dst = self._namer.fresh_temp("_tuple")
        self._emit(ContainerCreateNode(
            dst=dst, kind=ContainerKind.TUPLE, elements=elts, source_loc=loc,
        ))
        return dst

    def _compile_set(self, node: ast.Set, loc: SourceLocation) -> SSAVar:
        elts = [self._compile_expr(e) for e in node.elts]
        dst = self._namer.fresh_temp("_set")
        self._emit(ContainerCreateNode(
            dst=dst, kind=ContainerKind.SET, elements=elts, source_loc=loc,
        ))
        return dst

    # -- yield / yield from --------------------------------------------------

    def _compile_yield(self, node: ast.Yield, loc: SourceLocation) -> SSAVar:
        val = self._compile_expr(node.value) if node.value else None
        dst = self._namer.fresh_temp("_yield")
        self._emit(YieldNode(dst=dst, value=val, source_loc=loc))
        return dst

    def _compile_yield_from(self, node: ast.YieldFrom, loc: SourceLocation) -> SSAVar:
        val = self._compile_expr(node.value)
        dst = self._namer.fresh_temp("_yield")
        self._emit(YieldNode(dst=dst, value=val, is_yield_from=True, source_loc=loc))
        return dst

    # -- f-strings -----------------------------------------------------------

    def _compile_fstring(self, node: ast.JoinedStr, loc: SourceLocation) -> SSAVar:
        parts: List[SSAVar] = []
        static_parts: List[str] = []
        format_specs: Dict[int, str] = {}

        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                static_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                if len(static_parts) <= len(parts):
                    static_parts.append("")
                expr_var = self._compile_expr(value.value)
                parts.append(expr_var)
                if value.format_spec and isinstance(value.format_spec, ast.JoinedStr):
                    spec_parts = []
                    for sv in value.format_spec.values:
                        if isinstance(sv, ast.Constant):
                            spec_parts.append(str(sv.value))
                    if spec_parts:
                        format_specs[len(parts) - 1] = "".join(spec_parts)
            else:
                expr_var = self._compile_expr(value)
                if len(static_parts) <= len(parts):
                    static_parts.append("")
                parts.append(expr_var)

        dst = self._namer.fresh_temp("_fstr")
        self._emit(FormatStringNode(
            dst=dst,
            parts=parts,
            static_parts=static_parts,
            format_specs=format_specs,
            source_loc=loc,
        ))
        return dst

    # -- walrus operator (:=) ------------------------------------------------

    def _compile_named_expr(self, node: ast.NamedExpr, loc: SourceLocation) -> SSAVar:
        value = self._compile_expr(node.value)
        if isinstance(node.target, ast.Name):
            dst = self._namer.fresh_version(node.target.id)
            self._emit(AssignNode(dst=dst, src=value, source_loc=loc))
            return dst
        return value

    # -- slice expression ----------------------------------------------------

    def _compile_slice_expr(self, node: ast.Slice, loc: SourceLocation) -> SSAVar:
        lower = self._compile_expr(node.lower) if node.lower else None
        upper = self._compile_expr(node.upper) if node.upper else None
        step = self._compile_expr(node.step) if node.step else None
        dst = self._namer.fresh_temp("_slice")
        none_var = self._namer.fresh_temp("_none")
        self._emit(LiteralNode(
            dst=none_var, kind=LiteralKind.NONE, value=None, source_loc=loc,
        ))
        # represent as a call to slice()
        self._emit(CallNode(
            result=dst,
            func=SSAVar("slice"),
            args=[lower or none_var, upper or none_var, step or none_var],
            source_loc=loc,
        ))
        return dst


# ---------------------------------------------------------------------------
# Dominator tree & dominance frontier (for phi insertion)
# ---------------------------------------------------------------------------

class DominatorTree:
    """Computes dominators and dominance frontiers for phi-node placement."""

    def __init__(self, cfg: CFG) -> None:
        self._cfg = cfg
        self.idom: Dict[str, Optional[str]] = {}
        self.dom_frontier: Dict[str, Set[str]] = defaultdict(set)
        self._compute()

    def _compute(self) -> None:
        blocks = self._cfg.block_order
        if not blocks:
            return

        entry = blocks[0]
        self.idom = {b: None for b in blocks}
        self.idom[entry] = entry

        block_to_idx = {b: i for i, b in enumerate(blocks)}
        changed = True

        while changed:
            changed = False
            for b in blocks[1:]:
                preds = [
                    p for p in self._cfg.blocks[b].predecessors
                    if p in block_to_idx and self.idom.get(p) is not None
                ]
                if not preds:
                    continue
                new_idom = preds[0]
                for p in preds[1:]:
                    new_idom = self._intersect(
                        new_idom, p, block_to_idx,
                    )
                if self.idom[b] != new_idom:
                    self.idom[b] = new_idom
                    changed = True

        # dominance frontiers
        for b in blocks:
            preds = self._cfg.blocks[b].predecessors
            if len(preds) >= 2:
                for p in preds:
                    runner = p
                    while runner is not None and runner != self.idom.get(b):
                        self.dom_frontier[runner].add(b)
                        runner = self.idom.get(runner)

    def _intersect(
        self, b1: str, b2: str, idx: Dict[str, int],
    ) -> str:
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while idx.get(finger1, 0) > idx.get(finger2, 0):
                finger1 = self.idom.get(finger1, finger1)  # type: ignore[assignment]
            while idx.get(finger2, 0) > idx.get(finger1, 0):
                finger2 = self.idom.get(finger2, finger2)  # type: ignore[assignment]
        return finger1

    def dominance_frontier(self, block: str) -> Set[str]:
        return self.dom_frontier.get(block, set())


# ---------------------------------------------------------------------------
# PythonSSACompiler — main entry point
# ---------------------------------------------------------------------------

class PythonSSACompiler:
    """Main compiler: Python source → IRModule.

    Usage::

        compiler = PythonSSACompiler()
        ir_module = compiler.compile_source("x = 1\\nprint(x)")
        print(ir_module.pretty_print())
    """

    def __init__(self, source_file: str = "<unknown>") -> None:
        self._source_file = source_file

    def compile_source(self, source: str) -> IRModule:
        """Compile Python source code string into an IRModule."""
        tree = ast.parse(source, filename=self._source_file)
        return self.compile_module(tree)

    def compile_module(self, tree: ast.Module) -> IRModule:
        """Compile a parsed ast.Module into an IRModule."""
        module = IRModule(
            name=self._source_file,
            source_file=self._source_file,
        )

        # 1. Scope resolution
        resolver = ScopeResolver()
        scope_map = resolver.resolve(tree)

        # 2. Build SSA IR
        builder = SSABuilder(
            source_file=self._source_file,
            module=module,
        )
        builder._scope_map = scope_map

        builder._compile_body(tree.body)

        # ensure entry block has a terminator
        last_bb = builder._current_bb
        if not last_bb.terminator:
            none_var = builder._namer.fresh_temp("_none")
            builder._emit(LiteralNode(
                dst=none_var,
                kind=LiteralKind.NONE,
                value=None,
                source_loc=SourceLocation(file=self._source_file),
            ))
            builder._emit(ReturnNode(
                value=none_var,
                source_loc=SourceLocation(file=self._source_file),
            ))

        module.top_level_cfg = builder._cfg

        # 3. Compute dominance and insert remaining phis
        self._insert_dominance_phis(module)

        return module

    def _insert_dominance_phis(self, module: IRModule) -> None:
        """Insert phi nodes at dominance frontiers for all CFGs."""
        self._insert_phis_for_cfg(module.top_level_cfg)
        for fn in module.functions.values():
            self._insert_phis_for_cfg(fn.cfg)

    def _insert_phis_for_cfg(self, cfg: CFG) -> None:
        if len(cfg.blocks) < 2:
            return

        try:
            dom_tree = DominatorTree(cfg)
        except Exception:
            return

        # collect variable definitions per block
        defs_in_block: Dict[str, Set[str]] = defaultdict(set)
        for label, block in cfg.blocks.items():
            for node in block.nodes:
                for v in node.defined_vars:
                    if not v.name.startswith("_"):
                        defs_in_block[v.name].add(label)

        # iterated dominance frontier for phi placement
        for var_name, def_blocks in defs_in_block.items():
            worklist: Deque[str] = deque(def_blocks)
            visited: Set[str] = set()
            phi_placed: Set[str] = set()

            while worklist:
                block = worklist.popleft()
                for frontier_block in dom_tree.dominance_frontier(block):
                    if frontier_block not in phi_placed:
                        phi_placed.add(frontier_block)
                        # check if phi already exists for this variable
                        fb = cfg.blocks.get(frontier_block)
                        if fb is None:
                            continue
                        has_phi = any(
                            isinstance(n, PhiNode) and n.dst.name == var_name
                            for n in fb.nodes
                        )
                        if not has_phi and len(fb.predecessors) > 1:
                            incoming = {
                                p: SSAVar(var_name, 0) for p in fb.predecessors
                            }
                            phi = PhiNode(
                                dst=SSAVar(var_name, 0),
                                incoming=incoming,
                                source_loc=SourceLocation(file=cfg.entry),
                            )
                            fb.nodes.insert(0, phi)
                        if frontier_block not in visited:
                            visited.add(frontier_block)
                            worklist.append(frontier_block)
