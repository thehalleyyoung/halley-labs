"""Python version migration assistant.

Provides AST-based analysis and automated migration tools:
- Python version compatibility checking (targeting 3.12+)
- Automated syntax migration for modern Python
- Deprecated usage detection across Python versions
- Sync-to-async code conversion
- Type annotation migration from inference
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class CompatLevel(Enum):
    COMPATIBLE = "compatible"
    NEEDS_CHANGES = "needs_changes"
    INCOMPATIBLE = "incompatible"


class DeprecationSeverity(Enum):
    REMOVED = "removed"
    DEPRECATED = "deprecated"
    WARNING = "warning"


@dataclass
class CompatIssue:
    line: int
    message: str
    severity: str = "warning"
    fix_suggestion: Optional[str] = None
    python_version: str = ""

    def __str__(self) -> str:
        ver = f" (Python {self.python_version})" if self.python_version else ""
        return f"line {self.line}: {self.message}{ver}"


@dataclass
class CompatReport:
    source_version: str
    target_version: str
    level: CompatLevel
    issues: List[CompatIssue] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"Compatibility {self.source_version} -> {self.target_version}: "
            f"{self.level.value} ({len(self.issues)} issues)\n{self.summary}"
        )


@dataclass
class DeprecatedUsage:
    line: int
    feature: str
    deprecated_in: str
    removed_in: Optional[str] = None
    replacement: Optional[str] = None
    severity: DeprecationSeverity = DeprecationSeverity.DEPRECATED

    def __str__(self) -> str:
        removed = f", removed in {self.removed_in}" if self.removed_in else ""
        repl = f" -> use {self.replacement}" if self.replacement else ""
        return f"line {self.line}: {self.feature} (deprecated in {self.deprecated_in}{removed}){repl}"


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def _parse_source(source: str) -> Optional[ast.Module]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        obj = node.func
        while isinstance(obj, ast.Attribute):
            parts.append(obj.attr)
            obj = obj.value
        if isinstance(obj, ast.Name):
            parts.append(obj.id)
        return ".".join(reversed(parts))
    return ""


def _get_dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


# ---------------------------------------------------------------------------
# Deprecated features database
# ---------------------------------------------------------------------------

_DEPRECATED_IMPORTS: Dict[str, Dict[str, str]] = {
    "imp": {"deprecated_in": "3.4", "removed_in": "3.12", "replacement": "importlib"},
    "distutils": {"deprecated_in": "3.10", "removed_in": "3.12", "replacement": "setuptools"},
    "aifc": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "audioop": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "cgi": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "urllib.parse"},
    "cgitb": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "traceback"},
    "chunk": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "crypt": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "hashlib"},
    "imghdr": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "filetype"},
    "mailcap": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "msilib": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "nis": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "nntplib": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "ossaudiodev": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "pipes": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "subprocess"},
    "sndhdr": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "filetype"},
    "spwd": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "sunau": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "telnetlib": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "N/A"},
    "uu": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "base64"},
    "xdrlib": {"deprecated_in": "3.11", "removed_in": "3.13", "replacement": "struct"},
}

_DEPRECATED_CALLS: Dict[str, Dict[str, str]] = {
    "asyncio.get_event_loop": {
        "deprecated_in": "3.10", "replacement": "asyncio.get_running_loop()",
    },
    "typing.Optional": {
        "deprecated_in": "3.10", "replacement": "X | None (PEP 604)",
    },
    "typing.Union": {
        "deprecated_in": "3.10", "replacement": "X | Y (PEP 604)",
    },
    "typing.List": {
        "deprecated_in": "3.9", "replacement": "list[X] (PEP 585)",
    },
    "typing.Dict": {
        "deprecated_in": "3.9", "replacement": "dict[X, Y] (PEP 585)",
    },
    "typing.Tuple": {
        "deprecated_in": "3.9", "replacement": "tuple[X, ...] (PEP 585)",
    },
    "typing.Set": {
        "deprecated_in": "3.9", "replacement": "set[X] (PEP 585)",
    },
    "typing.FrozenSet": {
        "deprecated_in": "3.9", "replacement": "frozenset[X] (PEP 585)",
    },
    "typing.Type": {
        "deprecated_in": "3.9", "replacement": "type[X] (PEP 585)",
    },
}


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

class _CompatVisitor(ast.NodeVisitor):
    """Check for Python 3.12 compatibility issues."""

    def __init__(self, target_version: str) -> None:
        self.target_version = target_version
        self.issues: List[CompatIssue] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _DEPRECATED_IMPORTS:
                info = _DEPRECATED_IMPORTS[top]
                removed = info.get("removed_in", "")
                if removed and self._version_ge(self.target_version, removed):
                    self.issues.append(CompatIssue(
                        line=node.lineno,
                        message=f"Module '{top}' removed in Python {removed}",
                        severity="error",
                        fix_suggestion=f"Use {info.get('replacement', 'alternative')}",
                        python_version=removed,
                    ))
                else:
                    self.issues.append(CompatIssue(
                        line=node.lineno,
                        message=f"Module '{top}' deprecated since Python {info['deprecated_in']}",
                        severity="warning",
                        fix_suggestion=f"Use {info.get('replacement', 'alternative')}",
                        python_version=info["deprecated_in"],
                    ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            if top in _DEPRECATED_IMPORTS:
                info = _DEPRECATED_IMPORTS[top]
                self.issues.append(CompatIssue(
                    line=node.lineno,
                    message=f"Module '{top}' deprecated since Python {info['deprecated_in']}",
                    severity="warning",
                    fix_suggestion=f"Use {info.get('replacement', 'alternative')}",
                    python_version=info["deprecated_in"],
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        if name in _DEPRECATED_CALLS:
            info = _DEPRECATED_CALLS[name]
            self.issues.append(CompatIssue(
                line=node.lineno,
                message=f"'{name}' deprecated since Python {info['deprecated_in']}",
                severity="warning",
                fix_suggestion=f"Use {info.get('replacement', 'alternative')}",
                python_version=info["deprecated_in"],
            ))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Check for typing.List[X] etc.
        name = _get_dotted_name(node.value)
        if name in _DEPRECATED_CALLS:
            info = _DEPRECATED_CALLS[name]
            self.issues.append(CompatIssue(
                line=node.lineno,
                message=f"'{name}' deprecated since Python {info['deprecated_in']}",
                severity="warning",
                fix_suggestion=f"Use {info.get('replacement', 'alternative')}",
                python_version=info["deprecated_in"],
            ))
        self.generic_visit(node)

    @staticmethod
    def _version_ge(ver_a: str, ver_b: str) -> bool:
        def to_tuple(v: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in v.split(".") if x.isdigit())
        return to_tuple(ver_a) >= to_tuple(ver_b)


def check_python_compatibility(source: str, target_version: str = "3.12") -> CompatReport:
    """Check if Python source code is compatible with a target Python version.

    Detects deprecated/removed modules, deprecated API calls, and syntax
    that may not work in the target version.
    """
    tree = _parse_source(source)
    if tree is None:
        return CompatReport("unknown", target_version, CompatLevel.INCOMPATIBLE,
                            summary="Could not parse source")

    v = _CompatVisitor(target_version)
    v.visit(tree)

    if not v.issues:
        level = CompatLevel.COMPATIBLE
    elif any(i.severity == "error" for i in v.issues):
        level = CompatLevel.INCOMPATIBLE
    else:
        level = CompatLevel.NEEDS_CHANGES

    return CompatReport(
        source_version="detected",
        target_version=target_version,
        level=level,
        issues=v.issues,
        summary=f"{len(v.issues)} compatibility issues found",
    )


# ---------------------------------------------------------------------------
# Automated migration to 3.12
# ---------------------------------------------------------------------------

class _MigrationTransformer(ast.NodeTransformer):
    """Transform AST to be compatible with Python 3.12+."""

    def __init__(self) -> None:
        self.changes: List[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        if node.module and node.module.split(".")[0] in _DEPRECATED_IMPORTS:
            info = _DEPRECATED_IMPORTS[node.module.split(".")[0]]
            replacement = info.get("replacement", "")
            if replacement and replacement != "N/A":
                self.changes.append(
                    f"line {node.lineno}: replaced import from '{node.module}' with '{replacement}'"
                )
                node.module = replacement
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # typing.List[X] -> list[X], typing.Dict[K, V] -> dict[K, V], etc.
        name = _get_dotted_name(node.value)
        replacements = {
            "typing.List": "list",
            "typing.Dict": "dict",
            "typing.Tuple": "tuple",
            "typing.Set": "set",
            "typing.FrozenSet": "frozenset",
            "typing.Type": "type",
        }
        if name in replacements:
            self.changes.append(
                f"line {node.lineno}: replaced '{name}' with '{replacements[name]}'"
            )
            node.value = ast.Name(id=replacements[name], ctx=ast.Load())
            ast.fix_missing_locations(node)

        # typing.Optional[X] -> X | None
        if name == "typing.Optional":
            self.changes.append(f"line {node.lineno}: replaced 'Optional[X]' with 'X | None'")
            new_node = ast.BinOp(
                left=node.slice,
                op=ast.BitOr(),
                right=ast.Constant(value=None),
            )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        return self.generic_visit(node)


def migrate_to_3_12(source: str) -> str:
    """Auto-migrate Python source code to Python 3.12 syntax.

    Applies transformations:
    - Replace deprecated typing imports (List -> list, Dict -> dict, etc.)
    - Replace Optional[X] with X | None
    - Replace deprecated module imports
    """
    tree = _parse_source(source)
    if tree is None:
        return source

    transformer = _MigrationTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# Deprecated usage detection
# ---------------------------------------------------------------------------

class _DeprecatedVisitor(ast.NodeVisitor):
    """Find all deprecated features in source code."""

    def __init__(self) -> None:
        self.usages: List[DeprecatedUsage] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _DEPRECATED_IMPORTS:
                info = _DEPRECATED_IMPORTS[top]
                self.usages.append(DeprecatedUsage(
                    line=node.lineno,
                    feature=f"import {alias.name}",
                    deprecated_in=info["deprecated_in"],
                    removed_in=info.get("removed_in"),
                    replacement=info.get("replacement"),
                    severity=DeprecationSeverity.REMOVED if info.get("removed_in") else DeprecationSeverity.DEPRECATED,
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            if top in _DEPRECATED_IMPORTS:
                info = _DEPRECATED_IMPORTS[top]
                self.usages.append(DeprecatedUsage(
                    line=node.lineno,
                    feature=f"from {node.module} import ...",
                    deprecated_in=info["deprecated_in"],
                    removed_in=info.get("removed_in"),
                    replacement=info.get("replacement"),
                    severity=DeprecationSeverity.REMOVED if info.get("removed_in") else DeprecationSeverity.DEPRECATED,
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)

        # Check deprecated calls
        if name in _DEPRECATED_CALLS:
            info = _DEPRECATED_CALLS[name]
            self.usages.append(DeprecatedUsage(
                line=node.lineno,
                feature=name,
                deprecated_in=info["deprecated_in"],
                replacement=info.get("replacement"),
            ))

        # Check for string formatting with %
        if name == "format" and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
                self.usages.append(DeprecatedUsage(
                    line=node.lineno,
                    feature="str.format() with literal",
                    deprecated_in="style",
                    replacement="f-string",
                    severity=DeprecationSeverity.WARNING,
                ))

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Old-style string formatting: "..." % (...)
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            self.usages.append(DeprecatedUsage(
                line=node.lineno,
                feature="%-style string formatting",
                deprecated_in="style",
                replacement="f-string or str.format()",
                severity=DeprecationSeverity.WARNING,
            ))
        self.generic_visit(node)


def find_deprecated_usage(source: str) -> List[DeprecatedUsage]:
    """Find all deprecated Python features used in source code."""
    tree = _parse_source(source)
    if tree is None:
        return []
    v = _DeprecatedVisitor()
    v.visit(tree)
    return v.usages


# ---------------------------------------------------------------------------
# Async migration
# ---------------------------------------------------------------------------

class _AsyncTransformer(ast.NodeTransformer):
    """Convert synchronous functions to async equivalents."""

    # Calls that should become awaited
    _ASYNC_MAPPINGS: Dict[str, str] = {
        "requests.get": "aiohttp.ClientSession().get",
        "requests.post": "aiohttp.ClientSession().post",
        "requests.put": "aiohttp.ClientSession().put",
        "requests.delete": "aiohttp.ClientSession().delete",
        "requests.patch": "aiohttp.ClientSession().patch",
        "time.sleep": "asyncio.sleep",
        "open": "aiofiles.open",
    }

    _IO_CALLS = {"read", "write", "readline", "readlines", "send", "recv", "connect", "accept"}

    def __init__(self) -> None:
        self.changes: List[str] = []
        self._in_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Check if function contains any I/O or blocking calls
        if self._has_blocking_calls(node):
            self.changes.append(f"line {node.lineno}: converted '{node.name}' to async")
            new_node = ast.AsyncFunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            ast.copy_location(new_node, node)
            old = self._in_function
            self._in_function = True
            self.generic_visit(new_node)
            self._in_function = old
            return new_node

        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        name = _get_call_name(node)
        full_name = _get_call_name(node)

        # Replace known blocking calls
        if full_name in self._ASYNC_MAPPINGS:
            replacement = self._ASYNC_MAPPINGS[full_name]
            self.changes.append(f"line {node.lineno}: replaced '{full_name}' with '{replacement}'")
            parts = replacement.rsplit(".", 1)
            if len(parts) == 2:
                node.func = ast.Attribute(
                    value=ast.parse(parts[0], mode="eval").body,
                    attr=parts[1],
                    ctx=ast.Load(),
                )
            else:
                node.func = ast.Name(id=replacement, ctx=ast.Load())
            ast.fix_missing_locations(node)

        # Add await for I/O calls when in async function
        if self._in_function and name in self._IO_CALLS:
            await_node = ast.Await(value=node)
            ast.copy_location(await_node, node)
            return await_node

        return self.generic_visit(node)

    def _has_blocking_calls(self, node: ast.FunctionDef) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _get_call_name(child)
                full_name = _get_call_name(child)
                if full_name in self._ASYNC_MAPPINGS or name in self._IO_CALLS:
                    return True
        return False


def async_migration(source: str) -> str:
    """Convert synchronous Python code to async equivalents.

    - Converts functions with blocking I/O to async def
    - Replaces requests.* with aiohttp equivalents
    - Replaces time.sleep with asyncio.sleep
    - Adds await to I/O operations
    """
    tree = _parse_source(source)
    if tree is None:
        return source

    transformer = _AsyncTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# Type annotation migration
# ---------------------------------------------------------------------------

class _TypeInferenceVisitor(ast.NodeVisitor):
    """Infer types from assignments and usage patterns."""

    def __init__(self) -> None:
        self.inferred: Dict[str, Dict[str, str]] = {}  # func_name -> {param: type}
        self.return_types: Dict[str, str] = {}
        self._current_func: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._current_func = node.name
        self.inferred.setdefault(node.name, {})

        # Infer from default values
        defaults = node.args.defaults
        args = node.args.args
        offset = len(args) - len(defaults)
        for i, default in enumerate(defaults):
            arg = args[offset + i]
            if arg.annotation is None:
                inferred = self._infer_type_from_value(default)
                if inferred:
                    self.inferred[node.name][arg.arg] = inferred

        # Infer return type from return statements
        if node.returns is None:
            ret_type = self._infer_return_type(node)
            if ret_type:
                self.return_types[node.name] = ret_type

        self.generic_visit(node)
        self._current_func = None

    visit_AsyncFunctionDef = visit_FunctionDef

    def _infer_type_from_value(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, str):
                return "str"
            if isinstance(node.value, bytes):
                return "bytes"
            if node.value is None:
                return "None"
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Set):
            return "set"
        if isinstance(node, ast.Tuple):
            return "tuple"
        return None

    def _infer_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        return_types: Set[str] = set()
        has_return = False
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                has_return = True
                if child.value is not None:
                    t = self._infer_type_from_value(child.value)
                    if t:
                        return_types.add(t)
                else:
                    return_types.add("None")

        if not has_return:
            return "None"
        if len(return_types) == 1:
            return return_types.pop()
        if len(return_types) == 2 and "None" in return_types:
            other = (return_types - {"None"}).pop()
            return f"Optional[{other}]"
        return None


class _AnnotationTransformer(ast.NodeTransformer):
    """Add inferred type annotations to function signatures."""

    def __init__(self, inferred: Dict[str, Dict[str, str]], return_types: Dict[str, str]) -> None:
        self.inferred = inferred
        self.return_types = return_types
        self.changes: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        func_info = self.inferred.get(node.name, {})
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation is None and arg.arg in func_info:
                arg.annotation = ast.Name(id=func_info[arg.arg], ctx=ast.Load())
                self.changes.append(f"line {node.lineno}: annotated '{arg.arg}: {func_info[arg.arg]}'")

        if node.returns is None and node.name in self.return_types:
            ret = self.return_types[node.name]
            node.returns = ast.Name(id=ret, ctx=ast.Load())
            self.changes.append(f"line {node.lineno}: annotated return -> {ret}")

        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def type_annotation_migration(source: str) -> str:
    """Add type annotations to Python source code using inference.

    Infers types from:
    - Default parameter values
    - Return statement values
    - Common patterns (list/dict/set literals)

    Only adds annotations where none exist.
    """
    tree = _parse_source(source)
    if tree is None:
        return source

    # Phase 1: infer types
    inferrer = _TypeInferenceVisitor()
    inferrer.visit(tree)

    # Phase 2: apply annotations
    transformer = _AnnotationTransformer(inferrer.inferred, inferrer.return_types)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return source
