"""
Safe AST parsing and variable-tracking utilities.

Provides helper functions used by :class:`PandasAnalyzer` and
:class:`PySparkAnalyzer` for:

* Safe AST parsing with error recovery.
* Variable tracking through assignments.
* Method chain resolution (e.g. ``df.merge(...).groupby(...)``).
* Import resolution.
* Scope analysis.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_parse(source: str, filename: str = "<string>") -> ast.Module | None:
    """Parse Python source into an AST, returning *None* on failure.

    Parameters
    ----------
    source:
        Python source code.
    filename:
        Filename for error messages.

    Returns
    -------
    ast.Module or None
        The parsed AST, or *None* if parsing fails.
    """
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as exc:
        logger.warning("Failed to parse %s: %s", filename, exc)
        return None


def safe_parse_file(filepath: str) -> ast.Module | None:
    """Read and parse a Python file, returning *None* on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        return safe_parse(source, filepath)
    except (OSError, IOError) as exc:
        logger.warning("Failed to read %s: %s", filepath, exc)
        return None


def extract_imports(tree: ast.Module) -> dict[str, str]:
    """Extract import mappings from an AST.

    Returns a dict mapping local names to their fully-qualified module
    paths.  For example::

        import pandas as pd            → {"pd": "pandas"}
        from pyspark.sql import SparkSession → {"SparkSession": "pyspark.sql.SparkSession"}
    """
    imports: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = f"{module}.{alias.name}" if module else alias.name

    return imports


def track_assignments(tree: ast.Module) -> dict[str, list[int]]:
    """Track variable assignments, returning a mapping from name to line numbers.

    Parameters
    ----------
    tree:
        The AST to analyse.

    Returns
    -------
    dict[str, list[int]]
        Variable name → list of line numbers where it is assigned.
    """
    assignments: dict[str, list[int]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = _extract_names(target)
                for name in names:
                    assignments.setdefault(name, []).append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.target:
            names = _extract_names(node.target)
            for name in names:
                assignments.setdefault(name, []).append(node.lineno)
        elif isinstance(node, ast.AugAssign):
            names = _extract_names(node.target)
            for name in names:
                assignments.setdefault(name, []).append(node.lineno)

    return assignments


def resolve_method_chain(node: ast.expr) -> list[str]:
    """Resolve a chain of method calls into a list of method names.

    For example, ``df.merge(other).groupby("col").agg(...)`` →
    ``["merge", "groupby", "agg"]``.

    Parameters
    ----------
    node:
        The AST expression node.

    Returns
    -------
    list[str]
        Method names in call order (outermost first).
    """
    chain: list[str] = []
    current: ast.expr = node

    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                chain.append(current.func.attr)
                current = current.func.value
            else:
                break
        elif isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Subscript):
            chain.append("__getitem__")
            current = current.value
        else:
            break

    chain.reverse()
    return chain


def get_root_variable(node: ast.expr) -> str | None:
    """Get the root variable name from a chained expression.

    For ``df.merge(other).groupby("col")``, returns ``"df"``.
    """
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        elif isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                current = current.func.value
            elif isinstance(current.func, ast.Name):
                return current.func.id
            else:
                return None
        elif isinstance(current, ast.Subscript):
            current = current.value
        else:
            return None


def extract_string_args(call: ast.Call) -> list[str]:
    """Extract string literal arguments from a function call.

    Returns string values from positional arguments and keyword values.
    """
    strings: list[str] = []
    for arg in call.args:
        val = _extract_string_value(arg)
        if val is not None:
            strings.append(val)
    for kw in call.keywords:
        val = _extract_string_value(kw.value)
        if val is not None:
            strings.append(val)
    return strings


def extract_keyword_arg(call: ast.Call, keyword: str) -> ast.expr | None:
    """Extract a specific keyword argument from a call."""
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def extract_string_list_arg(node: ast.expr) -> list[str]:
    """Extract a list of string literals from an AST expression."""
    if isinstance(node, ast.List):
        result = []
        for elt in node.elts:
            val = _extract_string_value(elt)
            if val is not None:
                result.append(val)
        return result
    elif isinstance(node, ast.Tuple):
        result = []
        for elt in node.elts:
            val = _extract_string_value(elt)
            if val is not None:
                result.append(val)
        return result
    elif isinstance(node, ast.Constant):
        val = _extract_string_value(node)
        if val is not None:
            return [val]
    return []


def get_line_source(source: str, lineno: int) -> str:
    """Get the source text for a specific line number."""
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def find_function_calls(tree: ast.Module, func_name: str) -> list[ast.Call]:
    """Find all calls to a specific function name in the AST."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                calls.append(node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                calls.append(node)
    return calls


def find_method_calls(tree: ast.Module, method_name: str) -> list[ast.Call]:
    """Find all method calls (attribute calls) to a specific method."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == method_name:
                calls.append(node)
    return calls


def get_scope_variables(tree: ast.Module) -> dict[str, list[str]]:
    """Analyse scopes and return variables visible in each scope.

    Returns a mapping from scope name to list of variable names.
    The module-level scope is named ``"__module__"``.
    """
    scopes: dict[str, list[str]] = {"__module__": []}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope_name = node.name
            scope_vars: list[str] = [arg.arg for arg in node.args.args]
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        scope_vars.extend(_extract_names(target))
            scopes[scope_name] = scope_vars

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                scopes["__module__"].extend(_extract_names(target))

    return scopes


def ast_to_source(node: ast.expr) -> str:
    """Convert an AST node back to source code (best-effort)."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


# ── Private helpers ────────────────────────────────────────────────────

def _extract_names(target: ast.expr) -> list[str]:
    """Extract variable names from an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    elif isinstance(target, ast.Tuple):
        names = []
        for elt in target.elts:
            names.extend(_extract_names(elt))
        return names
    elif isinstance(target, ast.List):
        names = []
        for elt in target.elts:
            names.extend(_extract_names(elt))
        return names
    elif isinstance(target, ast.Starred):
        return _extract_names(target.value)
    elif isinstance(target, ast.Attribute):
        return [ast_to_source(target)]
    elif isinstance(target, ast.Subscript):
        return [ast_to_source(target)]
    return []


def _extract_string_value(node: ast.expr) -> str | None:
    """Extract a string value from an AST expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
