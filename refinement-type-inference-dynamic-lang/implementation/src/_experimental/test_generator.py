"""Automatic test generation from bug analysis results.

Generates:
- Regression tests (pytest) that reproduce detected bugs
- Property-based tests (hypothesis) for discovered functions
- Fuzz harnesses (atheris/pythonfuzz) for target functions
- Assertion suggestions (pre/post conditions) for functions
"""
from __future__ import annotations

import ast
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bug protocol — compatible with dataclasses from sibling modules
# ---------------------------------------------------------------------------

@dataclass
class Bug:
    """Minimal bug descriptor used for test generation."""
    kind: str  # e.g. "mutable_default", "resource_leak", "dead_code"
    message: str
    line: int
    column: int = 0
    function: Optional[str] = None
    variable: Optional[str] = None
    fix_suggestion: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indent(code: str, n: int = 4) -> str:
    return textwrap.indent(textwrap.dedent(code).strip(), " " * n)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def _extract_functions(source: str) -> List[ast.FunctionDef]:
    tree = ast.parse(source)
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _extract_classes(source: str) -> List[ast.ClassDef]:
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _get_function_args(func: ast.FunctionDef) -> List[Tuple[str, Optional[str]]]:
    """Return list of (param_name, annotation_string|None)."""
    result: List[Tuple[str, Optional[str]]] = []
    for arg in func.args.args:
        if arg.arg == "self":
            continue
        ann = ast.dump(arg.annotation) if arg.annotation else None
        if arg.annotation and isinstance(arg.annotation, ast.Name):
            ann = arg.annotation.id
        elif arg.annotation and isinstance(arg.annotation, ast.Constant):
            ann = repr(arg.annotation.value)
        result.append((arg.arg, ann))
    return result


def _has_return(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            return True
    return False


def _guess_strategy(ann: Optional[str]) -> str:
    """Map annotation name to a hypothesis strategy."""
    if ann is None:
        return "st.text()"
    mapping: Dict[str, str] = {
        "int": "st.integers()",
        "float": "st.floats(allow_nan=False)",
        "str": "st.text(max_size=100)",
        "bool": "st.booleans()",
        "bytes": "st.binary(max_size=100)",
        "list": "st.lists(st.integers())",
        "dict": "st.dictionaries(st.text(max_size=10), st.integers())",
        "set": "st.frozensets(st.integers())",
        "tuple": "st.tuples(st.integers())",
        "Optional": "st.one_of(st.none(), st.integers())",
    }
    for key, strat in mapping.items():
        if key in ann:
            return strat
    return "st.text()"


# ---------------------------------------------------------------------------
# Regression test generation
# ---------------------------------------------------------------------------

def generate_regression_tests(bugs: List[Bug]) -> str:
    """Generate pytest regression tests that exercise each reported bug.

    Each bug produces one test function that documents the issue and
    provides a skeleton test body.
    """
    lines: List[str] = [
        '"""Auto-generated regression tests for detected bugs."""',
        "import pytest",
        "",
        "",
    ]

    for i, bug in enumerate(bugs):
        func_name = f"test_bug_{i}_{_safe_name(bug.kind)}"
        lines.append(f"class Test{_safe_name(bug.kind).title().replace('_', '')}:")
        lines.append("")
        lines.append(f"    def {func_name}(self):")
        lines.append(f'        """Regression test for: {bug.message}')
        lines.append(f"        Detected at line {bug.line}, column {bug.column}.")
        if bug.fix_suggestion:
            lines.append(f"        Fix: {bug.fix_suggestion}")
        lines.append('        """')

        # Generate specific test body based on bug kind
        body = _generate_test_body(bug)
        for line in body:
            lines.append(f"        {line}")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def _generate_test_body(bug: Bug) -> List[str]:
    """Generate test body lines for a specific bug kind."""
    kind = bug.kind.lower()

    if "mutable_default" in kind:
        fn = bug.function or "target_function"
        param = bug.variable or "arg"
        return [
            f"# Mutable default argument in '{fn}' param '{param}'",
            f"# Calling twice should not share state",
            f"# result1 = {fn}()",
            f"# result2 = {fn}()",
            f"# assert result1 is not result2",
            "pytest.skip('fill in target function import')",
        ]

    if "resource_leak" in kind:
        var = bug.variable or "resource"
        return [
            f"# Resource '{var}' may not be properly closed",
            "# Verify the resource is closed after use:",
            f"# with open('test_file.txt', 'w') as {var}:",
            f"#     {var}.write('test')",
            f"# assert {var}.closed",
            "pytest.skip('fill in resource handling test')",
        ]

    if "dead_code" in kind:
        return [
            f"# Dead code detected at line {bug.line}",
            "# Verify the unreachable code path is intentional",
            "# or remove it",
            "pytest.skip('fill in dead code verification')",
        ]

    if "late_binding" in kind:
        var = bug.variable or "x"
        return [
            f"# Late binding: closure captures loop variable '{var}'",
            "# Each closure should capture its own value:",
            "funcs = []",
            "for i in range(5):",
            "    funcs.append(lambda i=i: i)  # fixed version",
            "results = [f() for f in funcs]",
            "assert results == [0, 1, 2, 3, 4]",
        ]

    if "exception" in kind or "swallow" in kind:
        return [
            "# Exception swallowing detected",
            "# Verify exceptions are properly handled:",
            "with pytest.raises(Exception):",
            "    raise ValueError('test error')",
        ]

    if "format" in kind:
        return [
            f"# String formatting bug: {bug.message}",
            "# Verify format string matches arguments:",
            "# result = 'template {}'.format('value')",
            "# assert 'value' in result",
            "pytest.skip('fill in format string test')",
        ]

    # Generic fallback
    return [
        f"# Bug: {bug.message}",
        f"# Kind: {bug.kind}",
        "pytest.skip('fill in regression test')",
    ]


# ---------------------------------------------------------------------------
# Property-based test generation
# ---------------------------------------------------------------------------

def generate_property_tests(source: str) -> str:
    """Generate hypothesis-based property tests for all functions in *source*."""
    functions = _extract_functions(source)
    if not functions:
        return "# No functions found to generate property tests for.\n"

    lines: List[str] = [
        '"""Auto-generated property-based tests using Hypothesis."""',
        "from hypothesis import given, strategies as st, assume, settings",
        "import pytest",
        "",
        "# Import the module under test:",
        "# from your_module import *",
        "",
        "",
    ]

    for func in functions:
        if isinstance(func, ast.AsyncFunctionDef):
            continue
        args = _get_function_args(func)
        if not args:
            continue

        # Build @given decorator
        given_args: List[str] = []
        param_list: List[str] = []
        for param_name, ann in args:
            strategy = _guess_strategy(ann)
            given_args.append(f"{param_name}={strategy}")
            param_list.append(param_name)

        has_ret = _has_return(func)

        lines.append(f"@given({', '.join(given_args)})")
        lines.append(f"@settings(max_examples=100)")
        lines.append(f"def test_{func.name}_properties({', '.join(param_list)}):")
        lines.append(f'    """Property test for {func.name}."""')

        # Generate property checks
        lines.append(f"    # Call the function under test")
        if has_ret:
            lines.append(f"    result = {func.name}({', '.join(param_list)})")
            lines.append(f"    # Add property assertions:")
            lines.append(f"    assert result is not None  # adjust as needed")

            # Type-specific properties
            if func.returns and isinstance(func.returns, ast.Name):
                ret_type = func.returns.id
                if ret_type == "int":
                    lines.append(f"    assert isinstance(result, int)")
                elif ret_type == "str":
                    lines.append(f"    assert isinstance(result, str)")
                elif ret_type == "list":
                    lines.append(f"    assert isinstance(result, list)")
                elif ret_type == "bool":
                    lines.append(f"    assert isinstance(result, bool)")
        else:
            lines.append(f"    # Function has no return value — test for no exceptions")
            lines.append(f"    {func.name}({', '.join(param_list)})")

        # Idempotence check for pure-looking functions
        if has_ret and not _looks_stateful(func):
            lines.append(f"    # Idempotence / determinism check:")
            lines.append(f"    result2 = {func.name}({', '.join(param_list)})")
            lines.append(f"    assert result == result2")

        lines.append("")
        lines.append("")

    return "\n".join(lines)


def _looks_stateful(func: ast.FunctionDef) -> bool:
    """Heuristic: does the function appear to mutate state?"""
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            return True
        if isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("append", "extend", "insert", "remove", "pop",
                                       "update", "add", "discard", "clear", "sort"):
                    return True
    return False


# ---------------------------------------------------------------------------
# Fuzz harness generation
# ---------------------------------------------------------------------------

def generate_fuzz_harness(function_source: str) -> str:
    """Generate an atheris/pythonfuzz fuzz harness for the given function."""
    functions = _extract_functions(function_source)
    if not functions:
        return "# No functions found to generate fuzz harness for.\n"

    func = functions[0]
    args = _get_function_args(func)

    lines: List[str] = [
        '"""Auto-generated fuzz harness using atheris."""',
        "import sys",
        "import atheris",
        "",
        "# Import the function under test:",
        f"# from your_module import {func.name}",
        "",
        "",
    ]

    # Generate data consumption based on parameter types
    lines.append("def fuzz_target(data):")
    lines.append('    """Fuzz target for atheris."""')
    lines.append("    fdp = atheris.FuzzedDataProvider(data)")

    consumed: List[str] = []
    for param_name, ann in args:
        consumer = _fuzz_consumer(ann)
        lines.append(f"    {param_name} = fdp.{consumer}")
        consumed.append(param_name)

    lines.append("    try:")
    lines.append(f"        {func.name}({', '.join(consumed)})")
    lines.append("    except (ValueError, TypeError, KeyError, IndexError, AttributeError):")
    lines.append("        pass  # expected exceptions")
    lines.append("    except Exception as e:")
    lines.append('        if "internal" in str(e).lower():')
    lines.append("            raise  # unexpected internal error")
    lines.append("")
    lines.append("")

    # Main block
    lines.append('if __name__ == "__main__":')
    lines.append("    atheris.Setup(sys.argv, fuzz_target)")
    lines.append("    atheris.Fuzz()")
    lines.append("")

    return "\n".join(lines)


def _fuzz_consumer(ann: Optional[str]) -> str:
    """Map annotation to atheris FuzzedDataProvider method."""
    if ann is None:
        return "ConsumeUnicodeNoSurrogates(100)"
    mapping: Dict[str, str] = {
        "int": "ConsumeInt(4)",
        "float": "ConsumeRegularFloat()",
        "str": "ConsumeUnicodeNoSurrogates(100)",
        "bool": "ConsumeBool()",
        "bytes": "ConsumeBytes(100)",
    }
    for key, consumer in mapping.items():
        if key in ann:
            return consumer
    return "ConsumeUnicodeNoSurrogates(100)"


# ---------------------------------------------------------------------------
# Assertion suggestion
# ---------------------------------------------------------------------------

def suggest_assertions(function_source: str) -> List[str]:
    """Suggest pre-conditions and post-conditions for a function.

    Returns a list of assertion strings suitable for insertion into
    the function body.
    """
    functions = _extract_functions(function_source)
    if not functions:
        return []

    func = functions[0]
    args = _get_function_args(func)
    suggestions: List[str] = []

    # Pre-conditions based on parameter annotations
    for param_name, ann in args:
        if ann:
            suggestions.append(f"assert isinstance({param_name}, {ann}), "
                               f"f\"expected {ann}, got {{type({param_name}).__name__}}\"")

        # Check for common patterns in the function body
        for node in ast.walk(func):
            # Division by parameter
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                right_names = _collect_names(node.right)
                if param_name in right_names:
                    suggestions.append(f"assert {param_name} != 0, \"{param_name} must not be zero\"")

            # Indexing by parameter
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Name) and node.slice.id == param_name:
                    if ann and "int" in ann:
                        suggestions.append(
                            f"assert {param_name} >= 0, \"{param_name} must be non-negative\""
                        )

    # Post-conditions based on return type annotation
    if func.returns:
        ret_ann = None
        if isinstance(func.returns, ast.Name):
            ret_ann = func.returns.id
        elif isinstance(func.returns, ast.Constant):
            ret_ann = repr(func.returns.value)

        if ret_ann:
            suggestions.append(f"# Post-condition: assert isinstance(result, {ret_ann})")

    # Check for None-dereference risk
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id
            # If parameter could be None
            for param_name, ann in args:
                if param_name == name and ann and "Optional" in ann:
                    suggestions.append(
                        f"assert {name} is not None, \"{name} must not be None\""
                    )

    # Check for len() calls suggesting non-empty requirements
    for node in ast.walk(func):
        if (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "len"):
            if node.left.args and isinstance(node.left.args[0], ast.Name):
                container = node.left.args[0].id
                for param_name, _ in args:
                    if param_name == container:
                        suggestions.append(
                            f"assert len({container}) > 0, \"{container} must not be empty\""
                        )

    # Deduplicate
    seen: Set[str] = set()
    unique: List[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def _collect_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names
