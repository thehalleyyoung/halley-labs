"""Advanced dataflow analysis for Python source code.

Provides AST-based dataflow tracking including:
- Variable tracing through assignments, augmented assignments, and function calls
- Unused variable detection (writes without subsequent reads)
- Dead code detection (unreachable statements after return/raise/break/continue)
- Taint analysis tracking data from sources to sinks
- Constant propagation to resolve statically-known values
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FlowKind(Enum):
    ASSIGNMENT = auto()
    AUGMENTED_ASSIGN = auto()
    FUNCTION_PARAM = auto()
    RETURN_VALUE = auto()
    FOR_TARGET = auto()
    WITH_TARGET = auto()
    IMPORT = auto()
    COMPREHENSION_TARGET = auto()
    GLOBAL = auto()
    NONLOCAL = auto()


class TaintState(Enum):
    TAINTED = "tainted"
    CLEAN = "clean"
    UNKNOWN = "unknown"


@dataclass
class DataflowPoint:
    """A single point in a variable's dataflow trace."""
    line: int
    column: int
    kind: FlowKind
    expression: str
    scope: str = "<module>"

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.name}] {self.expression} (in {self.scope})"


@dataclass
class DataflowTrace:
    """Full trace of a variable through the code."""
    variable: str
    definitions: List[DataflowPoint] = field(default_factory=list)
    uses: List[DataflowPoint] = field(default_factory=list)
    reassignments: List[DataflowPoint] = field(default_factory=list)

    @property
    def is_single_assignment(self) -> bool:
        return len(self.definitions) + len(self.reassignments) == 1


@dataclass
class UnusedVar:
    name: str
    line: int
    column: int
    scope: str = "<module>"
    reason: str = "assigned but never read"

    def __str__(self) -> str:
        return f"{self.line}:{self.column} unused variable '{self.name}' in {self.scope}: {self.reason}"


@dataclass
class DeadCode:
    line_start: int
    line_end: int
    reason: str
    scope: str = "<module>"

    def __str__(self) -> str:
        span = f"{self.line_start}-{self.line_end}" if self.line_start != self.line_end else str(self.line_start)
        return f"lines {span} in {self.scope}: {self.reason}"


@dataclass
class TaintFlow:
    source_line: int
    sink_line: int
    source_expr: str
    sink_expr: str
    variable: str
    path: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"tainted data flows from '{self.source_expr}' (line {self.source_line}) "
            f"to '{self.sink_expr}' (line {self.sink_line}) via '{self.variable}'"
        )


# ---------------------------------------------------------------------------
# Scope tracking
# ---------------------------------------------------------------------------

class _ScopeTracker(ast.NodeVisitor):
    """Collect variable definitions and uses per scope."""

    def __init__(self) -> None:
        self._scope_stack: List[str] = ["<module>"]
        self.defs: Dict[str, List[Tuple[str, int, int, FlowKind, str]]] = {}
        self.uses: Dict[str, List[Tuple[str, int, int]]] = {}

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    def _record_def(self, name: str, node: ast.AST, kind: FlowKind, expr: str = "") -> None:
        key = (self._scope, name)
        entry = (self._scope, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), kind, expr or name)
        self.defs.setdefault(name, []).append(entry)

    def _record_use(self, name: str, node: ast.AST) -> None:
        entry = (self._scope, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        self.uses.setdefault(name, []).append(entry)

    # --- visitors ---

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_def(node.name, node, FlowKind.ASSIGNMENT, f"def {node.name}(...)")
        self._scope_stack.append(node.name)
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self._record_def(arg.arg, arg, FlowKind.FUNCTION_PARAM, arg.arg)
        if node.args.vararg:
            self._record_def(node.args.vararg.arg, node.args.vararg, FlowKind.FUNCTION_PARAM, f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            self._record_def(node.args.kwarg.arg, node.args.kwarg, FlowKind.FUNCTION_PARAM, f"**{node.args.kwarg.arg}")
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_def(node.name, node, FlowKind.ASSIGNMENT, f"class {node.name}")
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name_node in _extract_names(target):
                self._record_def(name_node.id, name_node, FlowKind.ASSIGNMENT, ast.dump(node.value)[:60])
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target and isinstance(node.target, ast.Name):
            expr = ast.dump(node.value)[:60] if node.value else "<annotation-only>"
            self._record_def(node.target.id, node.target, FlowKind.ASSIGNMENT, expr)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._record_def(node.target.id, node.target, FlowKind.AUGMENTED_ASSIGN, ast.dump(node.value)[:60])
            self._record_use(node.target.id, node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        for name_node in _extract_names(node.target):
            self._record_def(name_node.id, name_node, FlowKind.FOR_TARGET, "for-target")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        for name_node in _extract_names(node.target):
            self._record_def(name_node.id, name_node, FlowKind.FOR_TARGET, "async-for-target")
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                for name_node in _extract_names(item.optional_vars):
                    self._record_def(name_node.id, name_node, FlowKind.WITH_TARGET, "with-target")
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._record_def(name, node, FlowKind.IMPORT, f"import {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._record_def(name, node, FlowKind.IMPORT, f"from {node.module} import {alias.name}")

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._record_def(name, node, FlowKind.GLOBAL, f"global {name}")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._record_def(name, node, FlowKind.NONLOCAL, f"nonlocal {name}")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record_use(node.id, node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comp(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comp(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comp(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comp(node)

    def _visit_comp(self, node: ast.AST) -> None:
        for gen in node.generators:  # type: ignore[attr-defined]
            for name_node in _extract_names(gen.target):
                self._record_def(name_node.id, name_node, FlowKind.COMPREHENSION_TARGET, "comp-target")
        self.generic_visit(node)


def _extract_names(node: ast.AST) -> List[ast.Name]:
    """Recursively extract all Name nodes from an assignment target."""
    if isinstance(node, ast.Name):
        return [node]
    if isinstance(node, (ast.Tuple, ast.List)):
        result: List[ast.Name] = []
        for elt in node.elts:
            result.extend(_extract_names(elt))
        return result
    if isinstance(node, ast.Starred):
        return _extract_names(node.value)
    return []


# ---------------------------------------------------------------------------
# Dead code detector
# ---------------------------------------------------------------------------

class _DeadCodeDetector(ast.NodeVisitor):
    """Detect unreachable code after terminator statements."""

    _TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def __init__(self) -> None:
        self.dead: List[DeadCode] = []
        self._scope_stack: List[str] = ["<module>"]

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    def _check_body(self, stmts: List[ast.stmt]) -> None:
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, self._TERMINATORS) and i < len(stmts) - 1:
                rest = stmts[i + 1:]
                start = rest[0].lineno
                end = rest[-1].end_lineno or rest[-1].lineno
                kind = type(stmt).__name__.lower()
                self.dead.append(DeadCode(
                    line_start=start,
                    line_end=end,
                    reason=f"unreachable code after '{kind}' on line {stmt.lineno}",
                    scope=self._scope,
                ))
                break

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope_stack.append(node.name)
        self._check_body(node.body)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        self._check_body(node.body)
        self._check_body(node.orelse)
        # Detect always-true / always-false branches
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                if node.orelse:
                    start = node.orelse[0].lineno
                    end = node.orelse[-1].end_lineno or node.orelse[-1].lineno
                    self.dead.append(DeadCode(start, end, "else branch never taken (condition is always True)", self._scope))
            else:
                start = node.body[0].lineno
                end = node.body[-1].end_lineno or node.body[-1].lineno
                self.dead.append(DeadCode(start, end, "if body never executed (condition is always False)", self._scope))
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_body(node.body)
        if isinstance(node.test, ast.Constant) and not node.test.value:
            start = node.body[0].lineno
            end = node.body[-1].end_lineno or node.body[-1].lineno
            self.dead.append(DeadCode(start, end, "while body never executed (condition is always False)", self._scope))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._check_body(node.body)
        for handler in node.handlers:
            self._check_body(handler.body)
        self._check_body(node.orelse)
        self._check_body(node.finalbody)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self._check_body(node.body)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Taint tracker
# ---------------------------------------------------------------------------

class _TaintTracker(ast.NodeVisitor):
    """Track tainted data from source calls to sink calls."""

    def __init__(self, sources: Set[str], sinks: Set[str]) -> None:
        self.sources = sources
        self.sinks = sinks
        self.tainted_vars: Dict[str, int] = {}  # var -> source line
        self.tainted_source_expr: Dict[str, str] = {}
        self.flows: List[TaintFlow] = []
        self._scope_stack: List[str] = ["<module>"]

    def _call_name(self, node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return None

    def _is_source(self, node: ast.Call) -> bool:
        name = self._call_name(node)
        return name is not None and name in self.sources

    def _is_sink(self, node: ast.Call) -> bool:
        name = self._call_name(node)
        return name is not None and name in self.sinks

    def _collect_names(self, node: ast.AST) -> Set[str]:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
        return names

    def visit_Assign(self, node: ast.Assign) -> None:
        # Check if RHS contains a source call
        is_source = False
        source_expr = ""
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call) and self._is_source(child):
                is_source = True
                source_expr = self._call_name(child) or "source"
                break

        if is_source:
            for target in node.targets:
                for name_node in _extract_names(target):
                    self.tainted_vars[name_node.id] = node.lineno
                    self.tainted_source_expr[name_node.id] = source_expr
        else:
            # Propagate taint: if any read var is tainted, written var becomes tainted
            read_names = self._collect_names(node.value)
            tainted_reads = read_names & set(self.tainted_vars)
            if tainted_reads:
                representative = next(iter(tainted_reads))
                for target in node.targets:
                    for name_node in _extract_names(target):
                        self.tainted_vars[name_node.id] = self.tainted_vars[representative]
                        self.tainted_source_expr[name_node.id] = self.tainted_source_expr.get(representative, "source")

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_sink(node):
            arg_names: Set[str] = set()
            for arg in node.args:
                arg_names |= self._collect_names(arg)
            for kw in node.keywords:
                if kw.value:
                    arg_names |= self._collect_names(kw.value)

            for name in arg_names:
                if name in self.tainted_vars:
                    sink_name = self._call_name(node) or "sink"
                    self.flows.append(TaintFlow(
                        source_line=self.tainted_vars[name],
                        sink_line=node.lineno,
                        source_expr=self.tainted_source_expr.get(name, "source"),
                        sink_expr=sink_name,
                        variable=name,
                    ))

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Constant propagation
# ---------------------------------------------------------------------------

class _ConstantPropagator(ast.NodeVisitor):
    """Simple forward constant propagation within module scope."""

    def __init__(self) -> None:
        self.constants: Dict[str, Any] = {}
        self._killed: Set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        val = self._eval_constant(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if val is not _UNKNOWN:
                    if target.id not in self._killed:
                        self.constants[target.id] = val
                else:
                    self._killed.add(target.id)
                    self.constants.pop(target.id, None)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            name = node.target.id
            if name in self.constants:
                rhs = self._eval_constant(node.value)
                if rhs is not _UNKNOWN:
                    try:
                        result = _apply_binop(node.op, self.constants[name], rhs)
                        if result is not _UNKNOWN:
                            self.constants[name] = result
                            return
                    except Exception:
                        pass
            self._killed.add(name)
            self.constants.pop(name, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Don't descend into function bodies for module-level propagation
        self.constants[node.name] = _UNKNOWN

    visit_AsyncFunctionDef = visit_FunctionDef

    def _eval_constant(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_constant(node.operand)
            if operand is _UNKNOWN:
                return _UNKNOWN
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
        if isinstance(node, ast.BinOp):
            left = self._eval_constant(node.left)
            right = self._eval_constant(node.right)
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            return _apply_binop(node.op, left, right)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elts = [self._eval_constant(e) for e in node.elts]
            if any(e is _UNKNOWN for e in elts):
                return _UNKNOWN
            if isinstance(node, ast.List):
                return elts
            if isinstance(node, ast.Tuple):
                return tuple(elts)
            return set(elts)
        return _UNKNOWN


_UNKNOWN = object()


def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
    try:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right if right != 0 else _UNKNOWN
        if isinstance(op, ast.FloorDiv):
            return left // right if right != 0 else _UNKNOWN
        if isinstance(op, ast.Mod):
            return left % right if right != 0 else _UNKNOWN
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
    except Exception:
        return _UNKNOWN
    return _UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse(source: str) -> ast.Module:
    return ast.parse(source, type_comments=True)


def trace_variable(source: str, var_name: str, line: int = 0) -> DataflowTrace:
    """Trace a variable through all definitions and uses in *source*.

    If *line* is given, only definitions at or near that line are included in
    ``definitions``; otherwise every definition is returned.
    """
    tree = _parse(source)
    tracker = _ScopeTracker()
    tracker.visit(tree)

    trace = DataflowTrace(variable=var_name)

    for scope, ln, col, kind, expr in tracker.defs.get(var_name, []):
        point = DataflowPoint(ln, col, kind, expr, scope)
        if kind in (FlowKind.AUGMENTED_ASSIGN,):
            trace.reassignments.append(point)
        elif line == 0 or abs(ln - line) <= 1:
            trace.definitions.append(point)
        else:
            trace.reassignments.append(point)

    for scope, ln, col in tracker.uses.get(var_name, []):
        trace.uses.append(DataflowPoint(ln, col, FlowKind.ASSIGNMENT, var_name, scope))

    return trace


def find_unused_variables(source: str) -> List[UnusedVar]:
    """Return variables that are assigned but never read."""
    tree = _parse(source)
    tracker = _ScopeTracker()
    tracker.visit(tree)

    # Build per-scope use sets
    use_names: Set[str] = set()
    for name, entries in tracker.uses.items():
        use_names.add(name)

    unused: List[UnusedVar] = []
    # Ignore conventional throwaway names
    _IGNORE = {"_", "__", "___", "self", "cls"}

    for name, def_entries in tracker.defs.items():
        if name in _IGNORE or name.startswith("_unused"):
            continue
        if name not in use_names:
            for scope, ln, col, kind, _ in def_entries:
                if kind == FlowKind.IMPORT:
                    reason = "imported but never used"
                elif kind == FlowKind.FUNCTION_PARAM:
                    reason = "parameter never used"
                else:
                    reason = "assigned but never read"
                unused.append(UnusedVar(name, ln, col, scope, reason))

    return unused


def find_dead_code(source: str) -> List[DeadCode]:
    """Find unreachable code after return/raise/break/continue."""
    tree = _parse(source)
    detector = _DeadCodeDetector()
    detector.visit(tree)
    return detector.dead


def taint_analysis(
    source: str,
    sources: List[str],
    sinks: List[str],
) -> List[TaintFlow]:
    """Track tainted data from *sources* to *sinks*.

    *sources* and *sinks* are function/method names (e.g. ``["input",
    "request.get"]``).  Returns every detected flow.
    """
    tree = _parse(source)
    tracker = _TaintTracker(set(sources), set(sinks))
    tracker.visit(tree)
    return tracker.flows


def constant_propagation(source: str) -> Dict[str, Any]:
    """Return a mapping of variable names to their statically-known constant values."""
    tree = _parse(source)
    propagator = _ConstantPropagator()
    propagator.visit(tree)
    return propagator.constants
