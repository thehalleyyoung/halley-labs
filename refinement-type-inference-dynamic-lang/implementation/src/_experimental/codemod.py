"""Automated code transformations (codemods) to fix detected bugs.

Provides AST-to-AST and text-based transformations:
- Apply fix suggestions from bug analysis
- Insert None/type guards before unsafe operations
- Insert bounds checks before indexing
- Modernize exception handling (bare except → specific catches)
- Preview fixes as unified diffs
"""
from __future__ import annotations

import ast
import copy
import difflib
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bug protocol
# ---------------------------------------------------------------------------

@dataclass
class Bug:
    """Minimal bug descriptor consumed by codemod functions."""
    kind: str
    message: str
    line: int
    column: int = 0
    function: Optional[str] = None
    variable: Optional[str] = None
    fix_suggestion: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse(source: str) -> ast.Module:
    return ast.parse(source, type_comments=True)


def _unparse(tree: ast.Module) -> str:
    return ast.unparse(tree)


def _source_lines(source: str) -> List[str]:
    return source.splitlines(keepends=True)


def _join_lines(lines: List[str]) -> str:
    return "".join(lines)


def _get_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


class _NodeLineMap:
    """Map AST nodes to their source lines for targeted edits."""

    def __init__(self, source: str) -> None:
        self.lines = _source_lines(source)
        self.tree = _parse(source)

    def get_line(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1]
        return ""


# ---------------------------------------------------------------------------
# Apply fixes
# ---------------------------------------------------------------------------

def apply_fixes(source: str, bugs: List[Bug]) -> str:
    """Apply all applicable automatic fixes to *source*.

    Processes bugs in reverse line order so that line numbers remain
    valid as edits are applied.
    """
    lines = _source_lines(source)
    # Sort bugs by line in reverse so edits don't shift later line numbers
    sorted_bugs = sorted(bugs, key=lambda b: b.line, reverse=True)

    for bug in sorted_bugs:
        kind = bug.kind.lower()
        idx = bug.line - 1
        if idx < 0 or idx >= len(lines):
            continue

        if "mutable_default" in kind:
            lines = _fix_mutable_default(lines, bug)
        elif "bare_except" in kind or "exception" in kind:
            lines = _fix_bare_except(lines, idx)
        elif "resource_leak" in kind:
            lines = _fix_resource_leak(lines, bug)
        elif "format" in kind and bug.fix_suggestion:
            # Generic fix_suggestion replacement
            pass

    return _join_lines(lines)


def _fix_mutable_default(lines: List[str], bug: Bug) -> List[str]:
    """Replace mutable default ``def f(x=[])`` with ``def f(x=None)`` + guard."""
    idx = bug.line - 1
    if idx < 0 or idx >= len(lines):
        return lines

    line = lines[idx]
    param = bug.variable or "arg"

    # Heuristic: find the default in the def line and replace with None
    # Handle list, dict, set defaults
    replacements = [
        (f"{param}=[]", f"{param}=None"),
        (f"{param}= []", f"{param}=None"),
        (f"{param} =[]", f"{param}=None"),
        (f"{param} = []", f"{param}=None"),
        (f"{param}={{}}", f"{param}=None"),
        (f"{param}= {{}}", f"{param}=None"),
        (f"{param} ={{}}", f"{param}=None"),
        (f"{param} = {{}}", f"{param}=None"),
    ]

    modified = False
    for old, new in replacements:
        if old in line:
            lines[idx] = line.replace(old, new, 1)
            modified = True
            break

    if modified:
        # Find the body start (next line after def) and insert guard
        body_idx = idx + 1
        while body_idx < len(lines) and lines[body_idx].strip() == "":
            body_idx += 1

        if body_idx < len(lines):
            indent = _get_indent(lines[body_idx])
            default_type = "[]" if "list" in (bug.message or "list") else "{}"
            guard_line = f"{indent}if {param} is None:\n"
            guard_assign = f"{indent}    {param} = {default_type}\n"
            lines.insert(body_idx, guard_assign)
            lines.insert(body_idx, guard_line)

    return lines


def _fix_bare_except(lines: List[str], idx: int) -> List[str]:
    """Replace ``except:`` with ``except Exception:``."""
    if idx < 0 or idx >= len(lines):
        return lines
    line = lines[idx]
    if "except:" in line:
        lines[idx] = line.replace("except:", "except Exception:", 1)
    return lines


def _fix_resource_leak(lines: List[str], bug: Bug) -> List[str]:
    """Wrap ``var = open(...)`` in a ``with`` statement."""
    idx = bug.line - 1
    if idx < 0 or idx >= len(lines):
        return lines

    line = lines[idx]
    var = bug.variable or "f"
    indent = _get_indent(line)
    stripped = line.lstrip()

    # Match pattern: var = open(...)
    if f"{var} = " in stripped and "open(" in stripped:
        # Extract the open(...) call
        eq_pos = stripped.index("=")
        rhs = stripped[eq_pos + 1:].strip().rstrip("\n")
        with_line = f"{indent}with {rhs} as {var}:\n"
        lines[idx] = with_line

        # Indent subsequent lines that use `var` until the next blank/unindented line
        j = idx + 1
        while j < len(lines):
            next_line = lines[j]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            next_indent = _get_indent(next_line)
            if len(next_indent) < len(indent) + 1 and next_stripped:
                # Same or less indentation — check if it uses the variable
                if var in next_stripped:
                    lines[j] = indent + "    " + next_stripped + "\n"
                    j += 1
                else:
                    break
            else:
                lines[j] = indent + "    " + next_stripped + "\n"
                j += 1

    return lines


# ---------------------------------------------------------------------------
# Add type guards
# ---------------------------------------------------------------------------

class _NoneDerefFinder(ast.NodeVisitor):
    """Find attribute accesses and calls on variables that may be None."""

    def __init__(self) -> None:
        self.guard_points: List[Tuple[int, str]] = []  # (line, var_name)
        self._optional_vars: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Identify Optional parameters
        for arg in node.args.args:
            if arg.annotation:
                ann_str = ast.dump(arg.annotation)
                if "None" in ann_str or "Optional" in ann_str:
                    self._optional_vars.add(arg.arg)

        # Check for assignments from functions that may return None
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                if isinstance(child.value, ast.Call):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            # Heuristic: .get(), .find(), re.search() etc may return None
                            if isinstance(child.value.func, ast.Attribute):
                                if child.value.func.attr in ("get", "find", "search", "match", "pop"):
                                    self._optional_vars.add(target.id)

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._optional_vars:
            self.guard_points.append((node.lineno, node.value.id))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self._optional_vars:
                self.guard_points.append((node.lineno, node.func.value.id))
        self.generic_visit(node)


def add_type_guards(source: str) -> str:
    """Insert ``if var is None`` checks before unsafe accesses on possibly-None variables."""
    tree = _parse(source)
    finder = _NoneDerefFinder()
    finder.visit(tree)

    if not finder.guard_points:
        return source

    lines = _source_lines(source)

    # Deduplicate and sort by line descending
    seen: Set[Tuple[int, str]] = set()
    unique_points: List[Tuple[int, str]] = []
    for point in finder.guard_points:
        if point not in seen:
            seen.add(point)
            unique_points.append(point)

    unique_points.sort(key=lambda p: p[0], reverse=True)

    for lineno, var_name in unique_points:
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            continue

        indent = _get_indent(lines[idx])
        guard = f"{indent}if {var_name} is None:\n"
        raise_line = f"{indent}    raise ValueError(f\"{var_name} must not be None\")\n"
        lines.insert(idx, raise_line)
        lines.insert(idx, guard)

    return _join_lines(lines)


# ---------------------------------------------------------------------------
# Add bounds checks
# ---------------------------------------------------------------------------

class _IndexFinder(ast.NodeVisitor):
    """Find subscript accesses that may be out of bounds."""

    def __init__(self) -> None:
        self.check_points: List[Tuple[int, str, str]] = []  # (line, container, index_expr)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load):
            container = node.value.id
            if isinstance(node.slice, ast.Name):
                self.check_points.append((node.lineno, container, node.slice.id))
            elif isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                self.check_points.append((node.lineno, container, repr(node.slice.value)))
            elif isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
                if isinstance(node.slice.operand, ast.Constant):
                    self.check_points.append((node.lineno, container, f"-{node.slice.operand.value}"))
        self.generic_visit(node)


def add_bounds_checks(source: str) -> str:
    """Insert bounds checks before list/sequence indexing operations."""
    tree = _parse(source)
    finder = _IndexFinder()
    finder.visit(tree)

    if not finder.check_points:
        return source

    lines = _source_lines(source)

    # Deduplicate and sort descending
    seen: Set[Tuple[int, str, str]] = set()
    unique: List[Tuple[int, str, str]] = []
    for pt in finder.check_points:
        if pt not in seen:
            seen.add(pt)
            unique.append(pt)
    unique.sort(key=lambda p: p[0], reverse=True)

    for lineno, container, index_expr in unique:
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            continue

        indent = _get_indent(lines[idx])
        # Use abs() for negative indices
        if index_expr.startswith("-"):
            check = (
                f"{indent}if abs({index_expr}) > len({container}):\n"
                f"{indent}    raise IndexError("
                f"f\"index {index_expr} out of range for {{len({container})}} elements\")\n"
            )
        else:
            try:
                int(index_expr)
                check = (
                    f"{indent}if {index_expr} >= len({container}):\n"
                    f"{indent}    raise IndexError("
                    f"f\"index {index_expr} out of range for {{len({container})}} elements\")\n"
                )
            except ValueError:
                check = (
                    f"{indent}if {index_expr} >= len({container}):\n"
                    f"{indent}    raise IndexError("
                    f"f\"index {{{index_expr}}} out of range for {{len({container})}} elements\")\n"
                )

        lines.insert(idx, check)

    return _join_lines(lines)


# ---------------------------------------------------------------------------
# Modernize exception handling
# ---------------------------------------------------------------------------

class _ExceptModernizer(ast.NodeVisitor):
    """Find bare except clauses and overly broad handlers."""

    def __init__(self) -> None:
        self.bare_excepts: List[int] = []  # line numbers
        self.broad_excepts: List[Tuple[int, str]] = []  # (line, current type)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.bare_excepts.append(node.lineno)
        elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
            self.broad_excepts.append((node.lineno, "BaseException"))
        self.generic_visit(node)


def modernize_exception_handling(source: str) -> str:
    """Fix bare ``except:`` clauses and add specific exception types.

    - ``except:`` → ``except Exception:``
    - ``except BaseException:`` → ``except Exception:``
    - Add ``as e`` binding and ``logger.exception()`` if handler is empty
    """
    tree = _parse(source)
    visitor = _ExceptModernizer()
    visitor.visit(tree)

    lines = _source_lines(source)

    # Process in reverse order
    all_fixes: List[Tuple[int, str]] = []
    for ln in visitor.bare_excepts:
        all_fixes.append((ln, "bare"))
    for ln, _ in visitor.broad_excepts:
        all_fixes.append((ln, "broad"))
    all_fixes.sort(key=lambda x: x[0], reverse=True)

    for lineno, fix_type in all_fixes:
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            continue

        line = lines[idx]

        if fix_type == "bare":
            if "except:" in line:
                lines[idx] = line.replace("except:", "except Exception as e:", 1)
                _maybe_add_logging(lines, idx)
        elif fix_type == "broad":
            if "BaseException" in line:
                lines[idx] = line.replace("BaseException", "Exception", 1)
                if " as " not in lines[idx]:
                    lines[idx] = lines[idx].replace("Exception:", "Exception as e:", 1)

    return _join_lines(lines)


def _maybe_add_logging(lines: List[str], except_idx: int) -> None:
    """If the handler body is just ``pass``, replace with logging."""
    body_idx = except_idx + 1
    if body_idx >= len(lines):
        return

    body_line = lines[body_idx].strip()
    if body_line == "pass":
        indent = _get_indent(lines[body_idx])
        lines[body_idx] = f"{indent}logger.exception(\"Unexpected error: %s\", e)\n"


# ---------------------------------------------------------------------------
# Preview fixes (unified diff)
# ---------------------------------------------------------------------------

def preview_fixes(source: str, bugs: List[Bug]) -> str:
    """Return a unified diff showing proposed fixes without applying them.

    This lets developers review changes before committing.
    """
    fixed = apply_fixes(source, bugs)

    original_lines = source.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile="original.py",
        tofile="fixed.py",
        lineterm="",
    )

    result = "".join(diff)
    if not result:
        return "# No changes to apply.\n"
    return result
