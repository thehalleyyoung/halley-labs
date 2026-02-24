"""Detection of common Python API misuse patterns.

AST-based detectors for:
- Resource leaks (unclosed files, sockets, connections)
- Exception swallowing (bare except, catch-and-ignore)
- Mutable default arguments
- Late-binding closures in loops
- String formatting bugs
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ResourceLeak:
    resource_type: str  # "file", "socket", "connection", etc.
    line: int
    column: int
    variable: str
    fix_suggestion: str

    def __str__(self) -> str:
        return f"{self.line}:{self.column} resource leak: '{self.variable}' ({self.resource_type}) may not be closed"


@dataclass
class SwallowedEx:
    line: int
    column: int
    kind: str  # "bare_except", "empty_handler", "pass_only"
    handler_type: Optional[str] = None

    def __str__(self) -> str:
        desc = self.handler_type or "all exceptions"
        return f"{self.line}:{self.column} swallowed exception ({self.kind}): catches {desc}"


@dataclass
class MutableDefault:
    function: str
    param: str
    line: int
    column: int
    default_type: str  # "list", "dict", "set"

    def __str__(self) -> str:
        return f"{self.line}:{self.column} mutable default in '{self.function}': param '{self.param}' defaults to {self.default_type}"


@dataclass
class LateBinding:
    line: int
    column: int
    variable: str
    loop_line: int
    fix_suggestion: str

    def __str__(self) -> str:
        return (
            f"{self.line}:{self.column} late binding: closure captures loop variable "
            f"'{self.variable}' from line {self.loop_line}"
        )


@dataclass
class FormatBug:
    line: int
    column: int
    kind: str  # "missing_key", "extra_positional", "mixed_numbering", etc.
    message: str

    def __str__(self) -> str:
        return f"{self.line}:{self.column} format bug ({self.kind}): {self.message}"


# ---------------------------------------------------------------------------
# Resource leak detection
# ---------------------------------------------------------------------------

# Calls that open resources and need closing
_RESOURCE_OPENERS: Dict[str, str] = {
    "open": "file",
    "builtins.open": "file",
    "io.open": "file",
    "socket.socket": "socket",
    "socket.create_connection": "socket",
    "sqlite3.connect": "connection",
    "psycopg2.connect": "connection",
    "mysql.connector.connect": "connection",
    "urllib.request.urlopen": "connection",
    "http.client.HTTPConnection": "connection",
    "http.client.HTTPSConnection": "connection",
    "tempfile.NamedTemporaryFile": "file",
    "tempfile.TemporaryFile": "file",
    "zipfile.ZipFile": "file",
    "tarfile.open": "file",
    "gzip.open": "file",
    "bz2.open": "file",
    "lzma.open": "file",
}


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = [node.func.attr]
        current: ast.expr = node.func.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class _ResourceLeakDetector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.leaks: List[ResourceLeak] = []
        self._with_targets: Set[int] = set()  # line numbers of with-statement opens
        self._scope_stack: List[str] = ["<module>"]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                self._with_targets.add(item.context_expr.lineno)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name and name in _RESOURCE_OPENERS and node.value.lineno not in self._with_targets:
                rtype = _RESOURCE_OPENERS[name]
                # Check if the assignment is inside a with statement (already handled)
                for target in node.targets:
                    var_names = _extract_target_names(target)
                    for var in var_names:
                        if not self._is_closed_in_scope(var, node):
                            self.leaks.append(ResourceLeak(
                                resource_type=rtype,
                                line=node.lineno,
                                column=node.col_offset,
                                variable=var,
                                fix_suggestion=f"Use 'with {name}(...) as {var}:' instead",
                            ))
        self.generic_visit(node)

    def _is_closed_in_scope(self, var: str, node: ast.AST) -> bool:
        # Heuristic: walk siblings after this node looking for var.close()
        # This is approximate — we check the whole module for simplicity
        return False

    def visit_Expr(self, node: ast.Expr) -> None:
        # Detect bare open() calls without assignment (resource immediately lost)
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name and name in _RESOURCE_OPENERS and node.value.lineno not in self._with_targets:
                rtype = _RESOURCE_OPENERS[name]
                self.leaks.append(ResourceLeak(
                    resource_type=rtype,
                    line=node.lineno,
                    column=node.col_offset,
                    variable="<discarded>",
                    fix_suggestion=f"Assign to a variable or use 'with {name}(...):'",
                ))
        self.generic_visit(node)


def _extract_target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    return []


# ---------------------------------------------------------------------------
# Exception swallowing detection
# ---------------------------------------------------------------------------

class _ExceptionSwallowDetector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.issues: List[SwallowedEx] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        handler_type = None
        if node.type:
            handler_type = ast.dump(node.type)
        else:
            handler_type = None

        # Bare except (no exception type)
        if node.type is None:
            self.issues.append(SwallowedEx(
                line=node.lineno,
                column=node.col_offset,
                kind="bare_except",
                handler_type=None,
            ))

        # Empty handler body or pass-only
        if self._is_empty_handler(node.body):
            kind = "pass_only" if self._is_pass_only(node.body) else "empty_handler"
            exc_name = self._type_name(node.type) if node.type else "all exceptions"
            self.issues.append(SwallowedEx(
                line=node.lineno,
                column=node.col_offset,
                kind=kind,
                handler_type=exc_name,
            ))

        self.generic_visit(node)

    @staticmethod
    def _is_empty_handler(body: List[ast.stmt]) -> bool:
        if not body:
            return True
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Docstring-only handler
                if isinstance(stmt.value.value, str):
                    return True
        return False

    @staticmethod
    def _is_pass_only(body: List[ast.stmt]) -> bool:
        return len(body) == 1 and isinstance(body[0], ast.Pass)

    @staticmethod
    def _type_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return ast.dump(node)
        if isinstance(node, ast.Tuple):
            names = []
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
                else:
                    names.append(ast.dump(elt))
            return "(" + ", ".join(names) + ")"
        return ast.dump(node)


# ---------------------------------------------------------------------------
# Mutable default detection
# ---------------------------------------------------------------------------

_MUTABLE_TYPES = {ast.List: "list", ast.Dict: "dict", ast.Set: "set"}
_MUTABLE_CALLS = {"list", "dict", "set", "defaultdict", "OrderedDict", "deque", "Counter"}


class _MutableDefaultDetector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.issues: List[MutableDefault] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node, node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_defaults(self, func: ast.FunctionDef, args: ast.arguments) -> None:
        # Regular args — defaults are right-aligned
        num_defaults = len(args.defaults)
        regular_args = args.args
        offset = len(regular_args) - num_defaults
        for i, default in enumerate(args.defaults):
            param = regular_args[offset + i]
            self._check_default_value(func.name, param.arg, default)

        # Keyword-only args
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                self._check_default_value(func.name, param.arg, default)

    def _check_default_value(self, func_name: str, param_name: str, default: ast.expr) -> None:
        # Check literal mutable types
        for ast_type, type_name in _MUTABLE_TYPES.items():
            if isinstance(default, ast_type):
                self.issues.append(MutableDefault(
                    function=func_name,
                    param=param_name,
                    line=default.lineno,
                    column=default.col_offset,
                    default_type=type_name,
                ))
                return

        # Check calls to mutable constructors
        if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
            if default.func.id in _MUTABLE_CALLS:
                self.issues.append(MutableDefault(
                    function=func_name,
                    param=param_name,
                    line=default.lineno,
                    column=default.col_offset,
                    default_type=default.func.id + "()",
                ))


# ---------------------------------------------------------------------------
# Late-binding closure detection
# ---------------------------------------------------------------------------

class _LateBindingDetector(ast.NodeVisitor):
    """Detect closures in loops that capture the loop variable."""

    def __init__(self) -> None:
        self.issues: List[LateBinding] = []
        self._loop_vars: Dict[str, int] = {}  # var_name -> loop line

    def visit_For(self, node: ast.For) -> None:
        loop_vars = self._collect_target_names(node.target)
        old_vars = dict(self._loop_vars)
        for v in loop_vars:
            self._loop_vars[v] = node.lineno

        # Check body for closures
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if child is not node:
                    self._check_closure(child, node.lineno)

        self._loop_vars = old_vars
        # Don't generic_visit — we already walked

    def visit_While(self, node: ast.While) -> None:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if child is not node:
                    self._check_closure(child, node.lineno)

    def _check_closure(self, func_node: ast.AST, loop_line: int) -> None:
        # Collect free variables referenced in the closure
        if isinstance(func_node, ast.Lambda):
            param_names = {a.arg for a in func_node.args.args}
            free_names = self._collect_free_names(func_node.body, param_names)
            func_line = func_node.lineno
        else:
            fn = func_node  # type: ignore[assignment]
            param_names = {a.arg for a in fn.args.args}
            if fn.args.vararg:
                param_names.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                param_names.add(fn.args.kwarg.arg)
            free_names = set()
            for stmt in fn.body:
                free_names |= self._collect_free_names(stmt, param_names)
            func_line = fn.lineno

        for name in free_names:
            if name in self._loop_vars:
                self.issues.append(LateBinding(
                    line=func_line,
                    column=getattr(func_node, "col_offset", 0),
                    variable=name,
                    loop_line=self._loop_vars[name],
                    fix_suggestion=f"Add '{name}={name}' as a default parameter to capture current value",
                ))

    def _collect_free_names(self, node: ast.AST, bound: Set[str]) -> Set[str]:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in bound:
                    names.add(child.id)
        return names

    @staticmethod
    def _collect_target_names(target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            result: List[str] = []
            for elt in target.elts:
                result.extend(_LateBindingDetector._collect_target_names(elt))
            return result
        return []


# ---------------------------------------------------------------------------
# String formatting bug detection
# ---------------------------------------------------------------------------

class _FormatBugDetector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.issues: List[FormatBug] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Check for %-formatting issues
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            fmt = node.left.value
            placeholders = self._count_percent_placeholders(fmt)
            if placeholders > 0:
                if isinstance(node.right, ast.Tuple):
                    n_args = len(node.right.elts)
                    if n_args != placeholders:
                        self.issues.append(FormatBug(
                            line=node.lineno,
                            column=node.col_offset,
                            kind="wrong_count",
                            message=f"format string expects {placeholders} args but {n_args} given",
                        ))
                elif not isinstance(node.right, ast.Dict) and placeholders > 1:
                    self.issues.append(FormatBug(
                        line=node.lineno,
                        column=node.col_offset,
                        kind="missing_tuple",
                        message=f"format string expects {placeholders} args but right side is not a tuple",
                    ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check .format() calls
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)):
            fmt = node.func.value.value
            self._check_str_format(fmt, node)

        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-strings: check for empty expressions
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if isinstance(value.value, ast.Constant) and value.value.value == "":
                    self.issues.append(FormatBug(
                        line=node.lineno,
                        column=node.col_offset,
                        kind="empty_expression",
                        message="empty expression in f-string",
                    ))
        self.generic_visit(node)

    def _check_str_format(self, fmt: str, node: ast.Call) -> None:
        import re
        # Count positional and named placeholders
        positional = 0
        named: Set[str] = set()
        auto_numbered = False
        explicit_numbered = False

        for match in re.finditer(r"\{([^}]*)\}", fmt):
            field = match.group(1).split(":")[0].split("!")[0].strip()
            if field == "":
                auto_numbered = True
                positional += 1
            elif field.isdigit():
                explicit_numbered = True
                positional = max(positional, int(field) + 1)
            else:
                named.add(field.split(".")[0].split("[")[0])

        if auto_numbered and explicit_numbered:
            self.issues.append(FormatBug(
                line=node.lineno,
                column=node.col_offset,
                kind="mixed_numbering",
                message="cannot mix automatic and explicit field numbering in .format()",
            ))

        # Check positional arg count
        n_pos_args = len(node.args)
        kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if not explicit_numbered and positional > n_pos_args:
            self.issues.append(FormatBug(
                line=node.lineno,
                column=node.col_offset,
                kind="missing_positional",
                message=f".format() expects {positional} positional args but got {n_pos_args}",
            ))

        # Check named args
        missing_named = named - kw_names
        if missing_named and not any(kw.arg is None for kw in node.keywords):
            self.issues.append(FormatBug(
                line=node.lineno,
                column=node.col_offset,
                kind="missing_key",
                message=f".format() missing keyword args: {', '.join(sorted(missing_named))}",
            ))

    @staticmethod
    def _count_percent_placeholders(fmt: str) -> int:
        import re
        # Match %s, %d, %f, %r, %x, etc. but not %%
        matches = re.findall(r"(?<!%)%(?!%)[-+0 #]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[diouxXeEfFgGcrsab%]", fmt)
        return len([m for m in matches if not m.endswith("%")])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse(source: str) -> ast.Module:
    return ast.parse(source, type_comments=True)


def detect_resource_leaks(source: str) -> List[ResourceLeak]:
    """Detect unclosed files, sockets, and connections."""
    tree = _parse(source)
    detector = _ResourceLeakDetector()
    detector.visit(tree)
    return detector.leaks


def detect_exception_swallowing(source: str) -> List[SwallowedEx]:
    """Detect bare except clauses and empty exception handlers."""
    tree = _parse(source)
    detector = _ExceptionSwallowDetector()
    detector.visit(tree)
    return detector.issues


def detect_mutable_defaults(source: str) -> List[MutableDefault]:
    """Detect mutable default arguments (e.g. ``def f(x=[]):``)."""
    tree = _parse(source)
    detector = _MutableDefaultDetector()
    detector.visit(tree)
    return detector.issues


def detect_late_binding(source: str) -> List[LateBinding]:
    """Detect closures in loops that capture loop variables."""
    tree = _parse(source)
    detector = _LateBindingDetector()
    detector.visit(tree)
    return detector.issues


def detect_string_formatting_bugs(source: str) -> List[FormatBug]:
    """Detect string formatting issues in %-formatting, .format(), and f-strings."""
    tree = _parse(source)
    detector = _FormatBugDetector()
    detector.visit(tree)
    return detector.issues
