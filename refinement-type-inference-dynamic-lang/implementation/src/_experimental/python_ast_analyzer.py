"""
Deep Python AST analysis: control flow graph, reaching definitions,
live variable analysis, constant propagation, and bug detection.
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class Location:
    file: str = "<unknown>"
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class Bug:
    category: str
    message: str
    location: Location
    severity: Severity = Severity.ERROR
    snippet: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.location}: {self.message}"


@dataclass
class Warning:
    category: str
    message: str
    location: Location
    severity: Severity = Severity.WARNING


@dataclass
class Metric:
    name: str
    value: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Control-flow graph
# ---------------------------------------------------------------------------

@dataclass
class CFGBlock:
    id: int
    stmts: List[ast.stmt] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)
    preds: List[int] = field(default_factory=list)
    label: str = ""
    is_entry: bool = False
    is_exit: bool = False


class CFG:
    """A simple control-flow graph over AST statements."""

    def __init__(self) -> None:
        self.blocks: Dict[int, CFGBlock] = {}
        self._next_id = 0
        self.entry: Optional[int] = None
        self.exit: Optional[int] = None

    def new_block(self, label: str = "") -> CFGBlock:
        bid = self._next_id
        self._next_id += 1
        blk = CFGBlock(id=bid, label=label)
        self.blocks[bid] = blk
        return blk

    def add_edge(self, src: int, dst: int) -> None:
        if dst not in self.blocks[src].succs:
            self.blocks[src].succs.append(dst)
        if src not in self.blocks[dst].preds:
            self.blocks[dst].preds.append(src)

    def predecessors(self, bid: int) -> List[int]:
        return self.blocks[bid].preds

    def successors(self, bid: int) -> List[int]:
        return self.blocks[bid].succs


class CFGBuilder(ast.NodeVisitor):
    """Build a CFG from a function body."""

    def __init__(self) -> None:
        self.cfg = CFG()
        self._current: Optional[CFGBlock] = None
        self._break_targets: List[CFGBlock] = []
        self._continue_targets: List[CFGBlock] = []

    def build(self, body: List[ast.stmt]) -> CFG:
        entry = self.cfg.new_block("entry")
        entry.is_entry = True
        self.cfg.entry = entry.id

        exit_blk = self.cfg.new_block("exit")
        exit_blk.is_exit = True
        self.cfg.exit = exit_blk.id

        self._current = entry
        self._visit_body(body)
        if self._current is not None:
            self.cfg.add_edge(self._current.id, exit_blk.id)
        return self.cfg

    # -- helpers ----------------------------------------------------------

    def _visit_body(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            if self._current is None:
                break
            self._visit_stmt(s)

    def _visit_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.If):
            self._visit_if(node)
        elif isinstance(node, (ast.For, ast.While)):
            self._visit_loop(node)
        elif isinstance(node, ast.Try):
            self._visit_try(node)
        elif isinstance(node, (ast.Return, ast.Raise)):
            self._visit_terminator(node)
        elif isinstance(node, ast.Break):
            if self._break_targets and self._current:
                self.cfg.add_edge(self._current.id, self._break_targets[-1].id)
            self._current = None
        elif isinstance(node, ast.Continue):
            if self._continue_targets and self._current:
                self.cfg.add_edge(self._current.id, self._continue_targets[-1].id)
            self._current = None
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if self._current is not None:
                self._current.stmts.append(node)
            self._visit_body(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if self._current is not None:
                self._current.stmts.append(node)
        else:
            if self._current is not None:
                self._current.stmts.append(node)

    def _visit_if(self, node: ast.If) -> None:
        assert self._current is not None
        self._current.stmts.append(node)
        cond_block = self._current

        then_block = self.cfg.new_block("if-then")
        self.cfg.add_edge(cond_block.id, then_block.id)
        self._current = then_block
        self._visit_body(node.body)
        then_end = self._current

        else_block = self.cfg.new_block("if-else")
        self.cfg.add_edge(cond_block.id, else_block.id)
        self._current = else_block
        if node.orelse:
            self._visit_body(node.orelse)
        else_end = self._current

        merge = self.cfg.new_block("if-merge")
        if then_end is not None:
            self.cfg.add_edge(then_end.id, merge.id)
        if else_end is not None:
            self.cfg.add_edge(else_end.id, merge.id)

        if then_end is not None or else_end is not None:
            self._current = merge
        else:
            self._current = None

    def _visit_loop(self, node: Union[ast.For, ast.While]) -> None:
        assert self._current is not None
        header = self.cfg.new_block("loop-header")
        self.cfg.add_edge(self._current.id, header.id)
        header.stmts.append(node)

        body_block = self.cfg.new_block("loop-body")
        self.cfg.add_edge(header.id, body_block.id)

        after = self.cfg.new_block("loop-after")

        self._break_targets.append(after)
        self._continue_targets.append(header)

        self._current = body_block
        self._visit_body(node.body)
        if self._current is not None:
            self.cfg.add_edge(self._current.id, header.id)

        self._break_targets.pop()
        self._continue_targets.pop()

        self.cfg.add_edge(header.id, after.id)

        if node.orelse:
            else_blk = self.cfg.new_block("loop-else")
            self.cfg.add_edge(header.id, else_blk.id)
            self._current = else_blk
            self._visit_body(node.orelse)
            if self._current is not None:
                self.cfg.add_edge(self._current.id, after.id)

        self._current = after

    def _visit_try(self, node: ast.Try) -> None:
        assert self._current is not None
        self._current.stmts.append(node)
        pre = self._current

        try_block = self.cfg.new_block("try-body")
        self.cfg.add_edge(pre.id, try_block.id)
        self._current = try_block
        self._visit_body(node.body)
        try_end = self._current

        after = self.cfg.new_block("try-after")

        for handler in node.handlers:
            h_blk = self.cfg.new_block("except")
            self.cfg.add_edge(pre.id, h_blk.id)
            self._current = h_blk
            self._visit_body(handler.body)
            if self._current is not None:
                self.cfg.add_edge(self._current.id, after.id)

        if try_end is not None:
            self.cfg.add_edge(try_end.id, after.id)

        if node.finalbody:
            fin_blk = self.cfg.new_block("finally")
            self.cfg.add_edge(after.id, fin_blk.id)
            self._current = fin_blk
            self._visit_body(node.finalbody)
        else:
            self._current = after

    def _visit_terminator(self, node: Union[ast.Return, ast.Raise]) -> None:
        if self._current is not None:
            self._current.stmts.append(node)
            if self.cfg.exit is not None:
                self.cfg.add_edge(self._current.id, self.cfg.exit)
        self._current = None


# ---------------------------------------------------------------------------
# Reaching definitions analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Definition:
    var: str
    block_id: int
    line: int


class ReachingDefinitions:
    """Forward data-flow: reaching definitions."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.gen: Dict[int, Set[Definition]] = {}
        self.kill: Dict[int, Set[Definition]] = {}
        self.reach_in: Dict[int, Set[Definition]] = {}
        self.reach_out: Dict[int, Set[Definition]] = {}
        self._all_defs: Dict[str, Set[Definition]] = {}

    def compute(self) -> None:
        for bid, blk in self.cfg.blocks.items():
            self.gen[bid] = set()
            self.kill[bid] = set()
            self.reach_in[bid] = set()
            self.reach_out[bid] = set()

        # Collect all definitions
        for bid, blk in self.cfg.blocks.items():
            for s in blk.stmts:
                for name in self._stmt_defs(s):
                    d = Definition(var=name, block_id=bid, line=getattr(s, "lineno", 0))
                    self.gen[bid].add(d)
                    self._all_defs.setdefault(name, set()).add(d)

        # Compute kill sets
        for bid in self.cfg.blocks:
            for d in self.gen[bid]:
                self.kill[bid] |= self._all_defs.get(d.var, set()) - {d}

        # Fixed-point iteration
        changed = True
        while changed:
            changed = False
            for bid in self.cfg.blocks:
                new_in: Set[Definition] = set()
                for p in self.cfg.predecessors(bid):
                    new_in |= self.reach_out[p]
                new_out = self.gen[bid] | (new_in - self.kill[bid])
                if new_out != self.reach_out[bid] or new_in != self.reach_in[bid]:
                    changed = True
                    self.reach_in[bid] = new_in
                    self.reach_out[bid] = new_out

    @staticmethod
    def _stmt_defs(node: ast.stmt) -> List[str]:
        names: List[str] = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                names.extend(ReachingDefinitions._target_names(t))
        elif isinstance(node, ast.AugAssign):
            names.extend(ReachingDefinitions._target_names(node.target))
        elif isinstance(node, (ast.AnnAssign,)):
            if node.target:
                names.extend(ReachingDefinitions._target_names(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.For):
            names.extend(ReachingDefinitions._target_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.append(alias.asname or alias.name)
        return names

    @staticmethod
    def _target_names(node: ast.expr) -> List[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            result: List[str] = []
            for elt in node.elts:
                result.extend(ReachingDefinitions._target_names(elt))
            return result
        if isinstance(node, ast.Starred):
            return ReachingDefinitions._target_names(node.value)
        return []


# ---------------------------------------------------------------------------
# Live variable analysis
# ---------------------------------------------------------------------------

class LiveVariables:
    """Backward data-flow: live variables at each block."""

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.use: Dict[int, Set[str]] = {}
        self.defs: Dict[int, Set[str]] = {}
        self.live_in: Dict[int, Set[str]] = {}
        self.live_out: Dict[int, Set[str]] = {}

    def compute(self) -> None:
        for bid, blk in self.cfg.blocks.items():
            self.use[bid] = set()
            self.defs[bid] = set()
            self.live_in[bid] = set()
            self.live_out[bid] = set()
            for s in blk.stmts:
                u = self._stmt_uses(s)
                d = set(ReachingDefinitions._stmt_defs(s))
                # uses before defs in this block
                self.use[bid] |= u - self.defs[bid]
                self.defs[bid] |= d

        changed = True
        while changed:
            changed = False
            for bid in self.cfg.blocks:
                new_out: Set[str] = set()
                for s in self.cfg.successors(bid):
                    new_out |= self.live_in[s]
                new_in = self.use[bid] | (new_out - self.defs[bid])
                if new_in != self.live_in[bid] or new_out != self.live_out[bid]:
                    changed = True
                    self.live_in[bid] = new_in
                    self.live_out[bid] = new_out

    def _stmt_uses(self, node: ast.stmt) -> Set[str]:
        collector = _NameCollector()
        if isinstance(node, ast.Assign):
            collector.visit(node.value)
        elif isinstance(node, ast.AugAssign):
            collector.visit(node.value)
            collector.visit(node.target)
        elif isinstance(node, ast.Return) and node.value:
            collector.visit(node.value)
        elif isinstance(node, ast.Expr):
            collector.visit(node.value)
        elif isinstance(node, ast.If):
            collector.visit(node.test)
        elif isinstance(node, (ast.For,)):
            collector.visit(node.iter)
        elif isinstance(node, ast.While):
            collector.visit(node.test)
        else:
            collector.visit(node)
        return collector.names


class _NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


# ---------------------------------------------------------------------------
# Constant propagation
# ---------------------------------------------------------------------------

class ConstantPropagation:
    """Forward data-flow: constant propagation over a CFG."""

    BOTTOM = object()
    TOP = object()

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.values_in: Dict[int, Dict[str, Any]] = {}
        self.values_out: Dict[int, Dict[str, Any]] = {}

    def compute(self) -> None:
        for bid in self.cfg.blocks:
            self.values_in[bid] = {}
            self.values_out[bid] = {}

        changed = True
        iterations = 0
        max_iter = 200
        while changed and iterations < max_iter:
            changed = False
            iterations += 1
            for bid, blk in self.cfg.blocks.items():
                # Meet over predecessors
                new_in: Dict[str, Any] = {}
                for p in self.cfg.predecessors(bid):
                    for var, val in self.values_out[p].items():
                        if var not in new_in:
                            new_in[var] = val
                        elif new_in[var] != val:
                            new_in[var] = self.TOP

                new_out = dict(new_in)
                for s in blk.stmts:
                    self._transfer(s, new_out)

                if new_out != self.values_out[bid]:
                    changed = True
                    self.values_in[bid] = new_in
                    self.values_out[bid] = new_out

    def _transfer(self, node: ast.stmt, env: Dict[str, Any]) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                val = self._eval_expr(node.value, env)
                env[target.id] = val
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                env[node.target.id] = self.TOP

    def _eval_expr(self, node: ast.expr, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return env.get(node.id, self.TOP)
        if isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left, env)
            right = self._eval_expr(node.right, env)
            if left is self.TOP or right is self.TOP:
                return self.TOP
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        return self.TOP
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    if right == 0:
                        return self.TOP
                    return left // right
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        return self.TOP
                    return left % right
            except Exception:
                return self.TOP
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand, env)
            if operand is self.TOP:
                return self.TOP
            try:
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.Not):
                    return not operand
            except Exception:
                return self.TOP
        return self.TOP

    def get_value(self, block_id: int, var: str) -> Any:
        return self.values_out.get(block_id, {}).get(var, self.TOP)


# ---------------------------------------------------------------------------
# Bug detectors
# ---------------------------------------------------------------------------

class _NoneDerefDetector(ast.NodeVisitor):
    """Detect attribute access / method call on potentially-None values."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._none_vars: Set[str] = set()
        self._scope_stack: List[Set[str]] = [set()]

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Name):
                if self._is_none_expr(node.value):
                    self._none_vars.add(t.id)
                elif t.id in self._none_vars:
                    self._none_vars.discard(t.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._none_vars:
            self.bugs.append(Bug(
                category="none-dereference",
                message=f"Attribute access '.{node.attr}' on '{node.value.id}' which may be None",
                location=Location(self.filename, node.lineno, node.col_offset),
                severity=Severity.ERROR,
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # After `if x is None: ...` the else branch knows x is not None
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            if isinstance(test.ops[0], ast.IsNot):
                if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                    if isinstance(test.left, ast.Name):
                        self._none_vars.discard(test.left.id)
        self.generic_visit(node)

    @staticmethod
    def _is_none_expr(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "get", "find", "search",
            ):
                return True
        return False


class _DivByZeroDetector(ast.NodeVisitor):
    """Detect division where the divisor could be zero."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._zero_vars: Set[str] = set()
        self._const_env: Dict[str, Any] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Name):
                if isinstance(node.value, ast.Constant) and node.value.value == 0:
                    self._zero_vars.add(t.id)
                    self._const_env[t.id] = 0
                elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                    self._zero_vars.discard(t.id)
                    self._const_env[t.id] = node.value.value
                else:
                    self._zero_vars.discard(t.id)
                    self._const_env.pop(t.id, None)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            divisor = node.right
            if isinstance(divisor, ast.Constant) and divisor.value == 0:
                self.bugs.append(Bug(
                    category="division-by-zero",
                    message="Division by literal zero",
                    location=Location(self.filename, node.lineno, node.col_offset),
                    severity=Severity.ERROR,
                ))
            elif isinstance(divisor, ast.Name) and divisor.id in self._zero_vars:
                self.bugs.append(Bug(
                    category="division-by-zero",
                    message=f"Division by '{divisor.id}' which is zero",
                    location=Location(self.filename, node.lineno, node.col_offset),
                    severity=Severity.ERROR,
                ))
        self.generic_visit(node)


class _IndexOutOfBoundsDetector(ast.NodeVisitor):
    """Detect list access with possibly-out-of-bounds index."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._known_lengths: Dict[str, int] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Name) and isinstance(node.value, ast.List):
                self._known_lengths[t.id] = len(node.value.elts)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant):
            name = node.value.id
            idx = node.slice.value
            if isinstance(idx, int) and name in self._known_lengths:
                length = self._known_lengths[name]
                if idx >= length or idx < -length:
                    self.bugs.append(Bug(
                        category="index-out-of-bounds",
                        message=f"Index {idx} out of bounds for '{name}' with length {length}",
                        location=Location(self.filename, node.lineno, node.col_offset),
                        severity=Severity.ERROR,
                    ))
        self.generic_visit(node)


class _TypeErrorDetector(ast.NodeVisitor):
    """Detect type errors: calling non-callable, invalid arithmetic operands."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._type_env: Dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self._type_env[t.id] = self._expr_type(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        lt = self._expr_type(node.left)
        rt = self._expr_type(node.right)
        if isinstance(node.op, ast.Add):
            if lt == "str" and rt == "int" or lt == "int" and rt == "str":
                self.bugs.append(Bug(
                    category="type-error",
                    message=f"Unsupported operand types for +: '{lt}' and '{rt}'",
                    location=Location(self.filename, node.lineno, node.col_offset),
                    severity=Severity.ERROR,
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            t = self._type_env.get(name)
            if t in ("int", "str", "float", "bool", "NoneType", "list"):
                self.bugs.append(Bug(
                    category="type-error",
                    message=f"'{name}' of type '{t}' is not callable",
                    location=Location(self.filename, node.lineno, node.col_offset),
                    severity=Severity.ERROR,
                ))
        self.generic_visit(node)

    def _expr_type(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return "int"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, str):
                return "str"
            if isinstance(node.value, bool):
                return "bool"
            if node.value is None:
                return "NoneType"
            return "unknown"
        if isinstance(node, ast.Name):
            return self._type_env.get(node.id, "unknown")
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.JoinedStr):
            return "str"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                lt = self._expr_type(node.left)
                rt = self._expr_type(node.right)
                if lt == "str" and rt == "str":
                    return "str"
                if lt == "int" and rt == "int":
                    return "int"
            return "unknown"
        return "unknown"


class _UninitializedVarDetector(ast.NodeVisitor):
    """Detect use of a variable before it is assigned on all paths."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._defined: Set[str] = set()
        self._builtins = {
            "print", "len", "range", "int", "str", "float", "bool", "list",
            "dict", "set", "tuple", "type", "isinstance", "issubclass",
            "enumerate", "zip", "map", "filter", "sorted", "reversed",
            "min", "max", "sum", "abs", "any", "all", "open", "input",
            "hasattr", "getattr", "setattr", "delattr", "super", "object",
            "True", "False", "None", "Exception", "ValueError", "TypeError",
            "KeyError", "IndexError", "AttributeError", "RuntimeError",
            "StopIteration", "NotImplementedError", "OSError", "IOError",
            "FileNotFoundError", "ImportError", "NameError",
            "__name__", "__file__", "__doc__",
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self._defined.copy()
        for arg in node.args.args:
            self._defined.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self._defined.add(arg.arg)
        if node.args.vararg:
            self._defined.add(node.args.vararg.arg)
        if node.args.kwarg:
            self._defined.add(node.args.kwarg.arg)
        self._defined.add(node.name)
        self.generic_visit(node)
        self._defined = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            for n in ReachingDefinitions._target_names(t):
                self._defined.add(n)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        for n in ReachingDefinitions._target_names(node.target):
            self._defined.add(n)

    def visit_For(self, node: ast.For) -> None:
        for n in ReachingDefinitions._target_names(node.target):
            self._defined.add(n)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._defined.add(alias.asname or alias.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._defined.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self._defined and node.id not in self._builtins:
                self.bugs.append(Bug(
                    category="uninitialized-variable",
                    message=f"Variable '{node.id}' may be used before assignment",
                    location=Location(self.filename, node.lineno, node.col_offset),
                    severity=Severity.WARNING,
                ))


class _UnreachableCodeDetector(ast.NodeVisitor):
    """Detect code after return/raise/break/continue."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []

    def _check_body(self, body: List[ast.stmt]) -> None:
        for i, s in enumerate(body):
            if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                if i + 1 < len(body):
                    next_s = body[i + 1]
                    self.bugs.append(Bug(
                        category="unreachable-code",
                        message=f"Unreachable code after {type(s).__name__.lower()}",
                        location=Location(self.filename, next_s.lineno, next_s.col_offset),
                        severity=Severity.WARNING,
                    ))
                break

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        self._check_body(node.body)
        self._check_body(node.orelse)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_body(node.body)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._check_body(node.body)
        for h in node.handlers:
            self._check_body(h.body)
        self._check_body(node.finalbody)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._check_body(node.body)
        self.generic_visit(node)


class _UnusedDetector(ast.NodeVisitor):
    """Detect unused variables and imports."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.bugs: List[Bug] = []
        self._assigned: Dict[str, Location] = {}
        self._imported: Dict[str, Location] = {}
        self._used: Set[str] = set()
        self._builtins = _UninitializedVarDetector(filename)._builtins

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Name) and not t.id.startswith("_"):
                self._assigned[t.id] = Location(self.filename, t.lineno, t.col_offset)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._imported[name] = Location(self.filename, node.lineno, node.col_offset)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            self._imported[name] = Location(self.filename, node.lineno, node.col_offset)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._used.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._used.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._used.add(node.name)
        self.generic_visit(node)

    def finalize(self) -> None:
        for name, loc in self._imported.items():
            if name not in self._used and not name.startswith("_"):
                self.bugs.append(Bug(
                    category="unused-import",
                    message=f"Unused import '{name}'",
                    location=loc,
                    severity=Severity.WARNING,
                ))
        for name, loc in self._assigned.items():
            if name not in self._used and name not in self._imported:
                self.bugs.append(Bug(
                    category="unused-variable",
                    message=f"Unused variable '{name}'",
                    location=loc,
                    severity=Severity.WARNING,
                ))


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    bugs: List[Bug] = field(default_factory=list)
    warnings: List[Warning] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    cfg: Optional[CFG] = None
    reaching_defs: Optional[ReachingDefinitions] = None
    live_vars: Optional[LiveVariables] = None
    constants: Optional[ConstantPropagation] = None

    @property
    def error_count(self) -> int:
        return sum(1 for b in self.bugs if b.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for b in self.bugs if b.severity == Severity.WARNING)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class PythonASTAnalyzer:
    """Analyse Python source code for bugs using AST-based checks."""

    def __init__(self, filename: str = "<string>") -> None:
        self.filename = filename

    def analyze_file(self, path: str) -> AnalysisResult:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        self.filename = path
        return self.analyze_source(source)

    def analyze_source(self, code: str) -> AnalysisResult:
        try:
            tree = ast.parse(code, filename=self.filename)
        except SyntaxError as e:
            return AnalysisResult(bugs=[Bug(
                category="syntax-error",
                message=str(e),
                location=Location(self.filename, e.lineno or 0, e.offset or 0),
                severity=Severity.ERROR,
            )])

        result = AnalysisResult()

        # Run bug detectors
        detectors: List[ast.NodeVisitor] = [
            _NoneDerefDetector(self.filename),
            _DivByZeroDetector(self.filename),
            _IndexOutOfBoundsDetector(self.filename),
            _TypeErrorDetector(self.filename),
            _UninitializedVarDetector(self.filename),
            _UnreachableCodeDetector(self.filename),
            _UnusedDetector(self.filename),
        ]

        for det in detectors:
            det.visit(tree)
            if hasattr(det, "finalize"):
                det.finalize()
            result.bugs.extend(getattr(det, "bugs", []))

        # Build CFG for top-level function bodies
        func_cfgs: Dict[str, CFG] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                builder = CFGBuilder()
                cfg = builder.build(node.body)
                func_cfgs[node.name] = cfg

                rd = ReachingDefinitions(cfg)
                rd.compute()

                lv = LiveVariables(cfg)
                lv.compute()

                cp = ConstantPropagation(cfg)
                cp.compute()

                if result.cfg is None:
                    result.cfg = cfg
                    result.reaching_defs = rd
                    result.live_vars = lv
                    result.constants = cp

        # Metrics
        lines = code.split("\n")
        result.metrics.append(Metric("total_lines", len(lines)))
        result.metrics.append(Metric("blank_lines", sum(1 for l in lines if not l.strip())))
        result.metrics.append(Metric("comment_lines", sum(1 for l in lines if l.strip().startswith("#"))))
        func_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        class_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        result.metrics.append(Metric("functions", func_count))
        result.metrics.append(Metric("classes", class_count))

        return result
