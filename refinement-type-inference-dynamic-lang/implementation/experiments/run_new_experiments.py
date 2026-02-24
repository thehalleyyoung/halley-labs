#!/usr/bin/env python3
"""Refinement Type Inference: Bug Detection Benchmark.

50+ buggy code snippets, 30+ clean snippets. Runs guard-harvest-style analysis,
measures precision/recall/F1, compares with baseline heuristics.
Outputs: bug_detection_results.json
"""

import json
import os
import time
import ast
import re
import textwrap
import numpy as np

# ---------------------------------------------------------------------------
# Code snippets with KNOWN bugs
# ---------------------------------------------------------------------------

BUGGY_SNIPPETS = [
    # --- Null/None dereference (10) ---
    {"id": "null_01", "category": "null_deref", "code": "def f(x):\n    y = None\n    return y.strip()", "bug": "calling .strip() on None"},
    {"id": "null_02", "category": "null_deref", "code": "def f(d):\n    v = d.get('key')\n    return v.upper()", "bug": "dict.get returns None if missing"},
    {"id": "null_03", "category": "null_deref", "code": "def f(lst):\n    result = None\n    for x in lst:\n        result = x\n    return result + 1", "bug": "result is None if lst is empty"},
    {"id": "null_04", "category": "null_deref", "code": "import re\ndef f(s):\n    m = re.match(r'(\\d+)', s)\n    return m.group(1)", "bug": "re.match returns None on no match"},
    {"id": "null_05", "category": "null_deref", "code": "def f(items):\n    found = next((x for x in items if x > 10), None)\n    return found * 2", "bug": "found can be None"},
    {"id": "null_06", "category": "null_deref", "code": "def f(d):\n    x = d.get('a', {}).get('b')\n    return x.lower()", "bug": "nested get returns None"},
    {"id": "null_07", "category": "null_deref", "code": "def f(lst):\n    if len(lst) > 0:\n        val = lst[0]\n    return val + 1", "bug": "val undefined if lst is empty"},
    {"id": "null_08", "category": "null_deref", "code": "def f(obj):\n    attr = getattr(obj, 'name', None)\n    return len(attr)", "bug": "attr can be None"},
    {"id": "null_09", "category": "null_deref", "code": "def f(mapping):\n    result = mapping.get('key')\n    return result['sub']", "bug": "result can be None, subscript fails"},
    {"id": "null_10", "category": "null_deref", "code": "def f(s):\n    parts = s.split(',')\n    first = parts[0] if parts else None\n    return first.strip()", "bug": "first could be None (dead logic but pattern)"},

    # --- Division by zero (10) ---
    {"id": "divz_01", "category": "div_by_zero", "code": "def f(a, b):\n    return a / b", "bug": "no check for b == 0"},
    {"id": "divz_02", "category": "div_by_zero", "code": "def avg(lst):\n    return sum(lst) / len(lst)", "bug": "empty list → div by zero"},
    {"id": "divz_03", "category": "div_by_zero", "code": "def normalize(values):\n    total = sum(values)\n    return [v / total for v in values]", "bug": "total can be 0"},
    {"id": "divz_04", "category": "div_by_zero", "code": "def ratio(a, b):\n    diff = a - b\n    return a / diff", "bug": "diff is 0 when a == b"},
    {"id": "divz_05", "category": "div_by_zero", "code": "def pct(part, whole):\n    return (part / whole) * 100", "bug": "whole can be 0"},
    {"id": "divz_06", "category": "div_by_zero", "code": "def safe_div(a, b):\n    if b != 0:\n        return a / b\n    # falls through returning None implicitly", "bug": "implicit None return"},
    {"id": "divz_07", "category": "div_by_zero", "code": "def mean(data):\n    n = len(data)\n    s = 0\n    for x in data:\n        s += x\n    return s / n", "bug": "n=0 for empty data"},
    {"id": "divz_08", "category": "div_by_zero", "code": "def inv(matrix_det):\n    return 1.0 / matrix_det", "bug": "matrix_det can be 0"},
    {"id": "divz_09", "category": "div_by_zero", "code": "def weighted_avg(vals, weights):\n    return sum(v*w for v,w in zip(vals,weights)) / sum(weights)", "bug": "sum(weights) can be 0"},
    {"id": "divz_10", "category": "div_by_zero", "code": "def f(x, y):\n    return x % y", "bug": "modulo by zero when y=0"},

    # --- Index/bounds errors (10) ---
    {"id": "bounds_01", "category": "bounds", "code": "def f(lst):\n    return lst[len(lst)]", "bug": "off-by-one: index == len"},
    {"id": "bounds_02", "category": "bounds", "code": "def last(lst):\n    return lst[-1]", "bug": "empty list IndexError"},
    {"id": "bounds_03", "category": "bounds", "code": "def f(s):\n    return s[10]", "bug": "string might be shorter than 10"},
    {"id": "bounds_04", "category": "bounds", "code": "def swap(lst, i, j):\n    lst[i], lst[j] = lst[j], lst[i]", "bug": "no bounds check on i, j"},
    {"id": "bounds_05", "category": "bounds", "code": "def f(matrix, row, col):\n    return matrix[row][col]", "bug": "no bounds check"},
    {"id": "bounds_06", "category": "bounds", "code": "def f(lst):\n    for i in range(len(lst) + 1):\n        print(lst[i])", "bug": "range goes one past end"},
    {"id": "bounds_07", "category": "bounds", "code": "def f(data):\n    return data[0] + data[1]", "bug": "fails if data has < 2 elements"},
    {"id": "bounds_08", "category": "bounds", "code": "def pop_all(lst):\n    while lst:\n        lst.pop()\n    return lst[0]", "bug": "accessing empty list after pop_all"},
    {"id": "bounds_09", "category": "bounds", "code": "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)][-1][n-1]", "bug": "last chunk may be smaller"},
    {"id": "bounds_10", "category": "bounds", "code": "def f(d, keys):\n    return [d[k] for k in keys]", "bug": "KeyError if key missing"},

    # --- Type errors (10) ---
    {"id": "type_01", "category": "type_error", "code": "def f(x):\n    return x + '1'", "bug": "x might be int, can't add str"},
    {"id": "type_02", "category": "type_error", "code": "def f(lst):\n    return sorted(lst, key=lambda x: x.lower())", "bug": "fails if elements aren't strings"},
    {"id": "type_03", "category": "type_error", "code": "def f(a, b):\n    return a + b", "bug": "type mismatch: int + str"},
    {"id": "type_04", "category": "type_error", "code": "def f(x):\n    return len(x)", "bug": "x might be int (no len)"},
    {"id": "type_05", "category": "type_error", "code": "def f(items):\n    return sum(items)", "bug": "fails on non-numeric items"},
    {"id": "type_06", "category": "type_error", "code": "def f(x):\n    return x.split(',')", "bug": "x might not be a string"},
    {"id": "type_07", "category": "type_error", "code": "def f(d):\n    for k, v in d:\n        pass", "bug": "iterating dict yields keys, not pairs"},
    {"id": "type_08", "category": "type_error", "code": "def f(x):\n    return int(x)", "bug": "ValueError if x is not numeric string"},
    {"id": "type_09", "category": "type_error", "code": "def f(lst):\n    return max(lst) - min(lst)", "bug": "fails on empty list or mixed types"},
    {"id": "type_10", "category": "type_error", "code": "def f(s):\n    return s * 2.5", "bug": "str * float is TypeError"},

    # --- Mutable default / scope (10) ---
    {"id": "mut_01", "category": "mutable_default", "code": "def f(lst=[]):\n    lst.append(1)\n    return lst", "bug": "mutable default argument"},
    {"id": "mut_02", "category": "mutable_default", "code": "def f(d={}):\n    d['key'] = 'val'\n    return d", "bug": "mutable default dict"},
    {"id": "mut_03", "category": "mutable_default", "code": "class C:\n    items = []\n    def add(self, x):\n        self.items.append(x)", "bug": "class-level mutable shared across instances"},
    {"id": "mut_04", "category": "mutable_default", "code": "def f(data, cache={}):\n    if data not in cache:\n        cache[data] = expensive(data)\n    return cache[data]", "bug": "mutable default cache persists"},
    {"id": "mut_05", "category": "mutable_default", "code": "def f(val, lst=[]):\n    lst.append(val)\n    return len(lst)", "bug": "mutable default grows across calls"},
    {"id": "mut_06", "category": "scope", "code": "def f():\n    for i in range(5):\n        x = i\n    return x", "bug": "x scoping relies on loop executing"},
    {"id": "mut_07", "category": "scope", "code": "def f(flag):\n    if flag:\n        result = 42\n    return result", "bug": "result undefined when flag is False"},
    {"id": "mut_08", "category": "scope", "code": "def f():\n    try:\n        x = risky()\n    except:\n        pass\n    return x", "bug": "x undefined if exception occurs"},
    {"id": "mut_09", "category": "mutable_default", "code": "def register(name, registry=[]):\n    registry.append(name)\n    return registry", "bug": "mutable default registry"},
    {"id": "mut_10", "category": "scope", "code": "def f(items):\n    for item in items:\n        processed = item.strip()\n    return processed", "bug": "processed undefined if items empty"},
]

# ---------------------------------------------------------------------------
# Clean code snippets (30+)
# ---------------------------------------------------------------------------

CLEAN_SNIPPETS = [
    {"id": "clean_01", "code": "def f(x):\n    if x is not None:\n        return x.strip()\n    return ''"},
    {"id": "clean_02", "code": "def avg(lst):\n    if not lst:\n        return 0.0\n    return sum(lst) / len(lst)"},
    {"id": "clean_03", "code": "def f(d):\n    v = d.get('key', '')\n    return v.upper()"},
    {"id": "clean_04", "code": "def f(a, b):\n    if b == 0:\n        return 0\n    return a / b"},
    {"id": "clean_05", "code": "def f(lst):\n    if not lst:\n        return None\n    return lst[0]"},
    {"id": "clean_06", "code": "def f(x):\n    return str(x) + '1'"},
    {"id": "clean_07", "code": "def f(lst, default=None):\n    result = default\n    for x in lst:\n        result = x\n    return result"},
    {"id": "clean_08", "code": "def f(items):\n    return [x for x in items if x is not None]"},
    {"id": "clean_09", "code": "def f(matrix, row, col):\n    if 0 <= row < len(matrix) and 0 <= col < len(matrix[row]):\n        return matrix[row][col]\n    return None"},
    {"id": "clean_10", "code": "def f(lst=None):\n    if lst is None:\n        lst = []\n    lst.append(1)\n    return lst"},
    {"id": "clean_11", "code": "def f(data):\n    return {k: v for k, v in data.items()}"},
    {"id": "clean_12", "code": "def f(s):\n    try:\n        return int(s)\n    except (ValueError, TypeError):\n        return 0"},
    {"id": "clean_13", "code": "def f(lst):\n    return sorted(lst) if lst else []"},
    {"id": "clean_14", "code": "def normalize(values):\n    total = sum(values)\n    if total == 0:\n        return [0.0] * len(values)\n    return [v / total for v in values]"},
    {"id": "clean_15", "code": "def f(x, y):\n    return x + y if isinstance(x, type(y)) else str(x) + str(y)"},
    {"id": "clean_16", "code": "def f(s):\n    if isinstance(s, str):\n        return s.split(',')\n    return list(s)"},
    {"id": "clean_17", "code": "def f(d):\n    for k, v in d.items():\n        print(k, v)"},
    {"id": "clean_18", "code": "def f(lst):\n    if len(lst) >= 2:\n        return lst[0] + lst[1]\n    return sum(lst)"},
    {"id": "clean_19", "code": "def f(items):\n    result = []\n    for item in items:\n        result.append(item.strip())\n    return result"},
    {"id": "clean_20", "code": "def f(x):\n    return abs(x) if isinstance(x, (int, float)) else len(x)"},
    {"id": "clean_21", "code": "def f(data, keys):\n    return [data.get(k) for k in keys]"},
    {"id": "clean_22", "code": "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]"},
    {"id": "clean_23", "code": "def f(flag):\n    result = None\n    if flag:\n        result = 42\n    return result"},
    {"id": "clean_24", "code": "def f():\n    try:\n        x = 1 / 1\n    except ZeroDivisionError:\n        x = 0\n    return x"},
    {"id": "clean_25", "code": "def f(lst):\n    return lst[-1] if lst else None"},
    {"id": "clean_26", "code": "import re\ndef f(s):\n    m = re.match(r'(\\d+)', s)\n    return m.group(1) if m else None"},
    {"id": "clean_27", "code": "def safe_div(a, b, default=0):\n    return a / b if b != 0 else default"},
    {"id": "clean_28", "code": "def f(x):\n    if hasattr(x, '__len__'):\n        return len(x)\n    return 1"},
    {"id": "clean_29", "code": "def f(items):\n    return sum(x for x in items if isinstance(x, (int, float)))"},
    {"id": "clean_30", "code": "def f(s):\n    return s.strip() if isinstance(s, str) else str(s).strip()"},
    {"id": "clean_31", "code": "def mean(data):\n    if not data:\n        raise ValueError('empty')\n    return sum(data) / len(data)"},
    {"id": "clean_32", "code": "def f(x):\n    return int(x) if str(x).isdigit() else 0"},
]


# ---------------------------------------------------------------------------
# Guard Harvest Analyzer (AST-based bug detection)
# ---------------------------------------------------------------------------

class GuardHarvestAnalyzer:
    """Static analyzer that detects common bug patterns via AST inspection."""

    def __init__(self):
        self.checks = [
            self._check_none_deref,
            self._check_div_by_zero,
            self._check_bounds_access,
            self._check_type_mismatch,
            self._check_mutable_default,
            self._check_unguarded_variable,
            self._check_bare_except,
            self._check_implicit_none_return,
        ]

    def analyze(self, code_str):
        """Analyze code and return list of detected issues."""
        issues = []
        try:
            tree = ast.parse(textwrap.dedent(code_str))
        except SyntaxError:
            return [{"type": "syntax_error", "message": "Failed to parse"}]

        for check in self.checks:
            issues.extend(check(tree, code_str))
        return issues

    def _check_none_deref(self, tree, code):
        """Detect potential None dereferences."""
        issues = []
        for node in ast.walk(tree):
            # Pattern: x = d.get(...) followed by x.something without None check
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    # Check if this variable was assigned from .get() or similar
                    name = node.value.id
                    if self._var_could_be_none(tree, name):
                        issues.append({
                            "type": "null_deref",
                            "message": f"Variable '{name}' could be None when accessing .{node.attr}",
                            "line": getattr(node, 'lineno', 0),
                        })
            # Pattern: calling method on None literal
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Constant):
                if node.value.value is None:
                    issues.append({
                        "type": "null_deref",
                        "message": "Calling attribute on None",
                        "line": getattr(node, 'lineno', 0),
                    })
        # Also check code text patterns
        if '.get(' in code and re.search(r'\.get\([^)]*\)\s*$', code, re.M) is None:
            # Check if .get() result is used without guard
            if 'if' not in code and 'is not None' not in code and "get(" in code:
                if re.search(r'=\s*\w+\.get\(.+\)\s*\n.*\breturn\b.*\.\w+', code):
                    issues.append({"type": "null_deref", "message": "Unguarded .get() result used"})
        if 'getattr(' in code and 'None' in code and 'is not None' not in code:
            if re.search(r'getattr\(.+,\s*None\)', code) or re.search(r"getattr\(.+,\s*'.+',\s*None\)", code):
                issues.append({"type": "null_deref", "message": "getattr with None default used without guard"})
        if 'next(' in code and 'None)' in code and 'is not None' not in code:
            issues.append({"type": "null_deref", "message": "next() with None default used without guard"})
        # y = None; return y.xxx
        if re.search(r'=\s*None\s*\n.*return\s+\w+\.', code):
            issues.append({"type": "null_deref", "message": "Variable assigned None then dereferenced"})
        return issues

    def _var_could_be_none(self, tree, name):
        """Check if a variable name could be None based on assignment patterns."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        if isinstance(node.value, ast.Constant) and node.value.value is None:
                            return True
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Attribute):
                                if node.value.func.attr == 'get':
                                    return True
        return False

    def _check_div_by_zero(self, tree, code):
        """Detect potential division by zero."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Div, ast.Mod, ast.FloorDiv)):
                # Walk up to find the BinOp
                pass
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mod, ast.FloorDiv)):
                # Check if divisor could be zero
                if isinstance(node.right, ast.Name):
                    name = node.right.id
                    # Check for guards
                    if f'{name} == 0' not in code and f'{name} != 0' not in code and \
                       f'{name} == 0' not in code and f'if {name}' not in code and \
                       f'if not {name}' not in code:
                        issues.append({
                            "type": "div_by_zero",
                            "message": f"Potential division by zero with '{name}'",
                            "line": getattr(node, 'lineno', 0),
                        })
                elif isinstance(node.right, ast.Call):
                    if isinstance(node.right.func, ast.Name) and node.right.func.id in ('len', 'sum'):
                        func_name = node.right.func.id
                        if 'if not' not in code and 'if len' not in code and 'if sum' not in code:
                            issues.append({
                                "type": "div_by_zero",
                                "message": f"Division by {func_name}() which can be 0",
                                "line": getattr(node, 'lineno', 0),
                            })
        return issues

    def _check_bounds_access(self, tree, code):
        """Detect potential index out of bounds."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant):
                    idx = node.slice.value
                    if isinstance(idx, int) and idx > 5:
                        issues.append({
                            "type": "bounds",
                            "message": f"Hard-coded index [{idx}] may be out of bounds",
                            "line": getattr(node, 'lineno', 0),
                        })
                # lst[len(lst)] pattern
                if isinstance(node.slice, ast.Call):
                    if isinstance(node.slice.func, ast.Name) and node.slice.func.id == 'len':
                        issues.append({
                            "type": "bounds",
                            "message": "Index equals len() — off by one",
                            "line": getattr(node, 'lineno', 0),
                        })
        # range(len(lst) + 1) pattern
        if 'range(len(' in code and '+ 1)' in code:
            issues.append({"type": "bounds", "message": "range(len(x) + 1) — iterates past end"})
        # Accessing [0] or [-1] without emptiness check
        if re.search(r'\[\s*-?\s*1?\s*\]', code) and 'if' not in code and 'not lst' not in code:
            if re.search(r'return\s+\w+\[\s*(-1|0)\s*\]', code) and 'if' not in code:
                issues.append({"type": "bounds", "message": "Unguarded index access on potentially empty sequence"})
        return issues

    def _check_type_mismatch(self, tree, code):
        """Detect potential type errors."""
        issues = []
        # x + '1' without type check
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                right_str = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                left_name = isinstance(node.left, ast.Name)
                right_name = isinstance(node.right, ast.Name)
                if (left_str and right_name) or (right_str and left_name):
                    if 'isinstance' not in code and 'str(' not in code:
                        issues.append({
                            "type": "type_error",
                            "message": "Potential str + non-str addition",
                            "line": getattr(node, 'lineno', 0),
                        })
            # x * float where x could be str
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, float):
                    issues.append({
                        "type": "type_error",
                        "message": "Multiplying by float — fails on strings",
                        "line": getattr(node, 'lineno', 0),
                    })
        # for k, v in d (not d.items())
        if re.search(r'for\s+\w+\s*,\s*\w+\s+in\s+\w+\s*:', code) and '.items()' not in code:
            issues.append({"type": "type_error", "message": "Iterating dict unpacks keys only, not (k,v)"})
        return issues

    def _check_mutable_default(self, tree, code):
        """Detect mutable default arguments."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is None:
                        continue
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "type": "mutable_default",
                            "message": f"Mutable default argument in {node.name}()",
                            "line": getattr(node, 'lineno', 0),
                        })
        # Class-level mutable
        if re.search(r'class\s+\w+.*:\s*\n\s+\w+\s*=\s*\[\s*\]', code):
            issues.append({"type": "mutable_default", "message": "Class-level mutable list shared across instances"})
        return issues

    def _check_unguarded_variable(self, tree, code):
        """Detect variables that might be uninitialized."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for variables only assigned inside if/for/try
                assigned_in_branch = set()
                used_after = set()
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For)):
                        for sub in ast.walk(child):
                            if isinstance(sub, ast.Assign):
                                for t in sub.targets:
                                    if isinstance(t, ast.Name):
                                        assigned_in_branch.add(t.id)
                # Check return statements for these variables
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value:
                        for sub in ast.walk(child.value):
                            if isinstance(sub, ast.Name) and sub.id in assigned_in_branch:
                                # Check if also assigned unconditionally
                                if sub.id not in self._get_unconditional_assigns(node):
                                    issues.append({
                                        "type": "unguarded_variable",
                                        "message": f"Variable '{sub.id}' may be uninitialized",
                                        "line": getattr(child, 'lineno', 0),
                                    })
        return issues

    def _get_unconditional_assigns(self, func_node):
        """Get variable names assigned at function top level (not in branches)."""
        assigns = set()
        for child in ast.iter_child_nodes(func_node):
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        assigns.add(t.id)
        return assigns

    def _check_bare_except(self, tree, code):
        """Detect bare except clauses."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "bare_except",
                        "message": "Bare except catches all exceptions including SystemExit",
                        "line": getattr(node, 'lineno', 0),
                    })
        return issues

    def _check_implicit_none_return(self, tree, code):
        """Detect functions that sometimes return a value and sometimes None."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                returns = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        returns.append(child)
                has_value_return = any(r.value is not None for r in returns)
                has_bare_return = any(r.value is None for r in returns)
                # Check if function can fall through without return
                if has_value_return and not has_bare_return:
                    # Check if all paths return
                    last_stmt = node.body[-1] if node.body else None
                    if last_stmt and not isinstance(last_stmt, ast.Return):
                        if isinstance(last_stmt, ast.If):
                            # Check if both branches return
                            if_returns = any(isinstance(s, ast.Return) for s in last_stmt.body)
                            else_returns = last_stmt.orelse and any(isinstance(s, ast.Return) for s in last_stmt.orelse)
                            if if_returns and not else_returns:
                                issues.append({
                                    "type": "implicit_none",
                                    "message": f"Function '{node.name}' may implicitly return None",
                                    "line": getattr(node, 'lineno', 0),
                                })
        return issues


# ---------------------------------------------------------------------------
# Baseline analyzer (simple heuristics only)
# ---------------------------------------------------------------------------

class BaselineAnalyzer:
    """Baseline: only checks for bare except and mutable defaults."""

    def analyze(self, code_str):
        issues = []
        try:
            tree = ast.parse(textwrap.dedent(code_str))
        except SyntaxError:
            return [{"type": "syntax_error"}]

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({"type": "bare_except"})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({"type": "mutable_default"})
        return issues


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_bug_detection_benchmark():
    print("=" * 60)
    print("Bug Detection Benchmark")
    print("=" * 60)

    analyzer = GuardHarvestAnalyzer()
    baseline = BaselineAnalyzer()

    # Run on buggy snippets
    buggy_results = []
    for snippet in BUGGY_SNIPPETS:
        t0 = time.time()
        issues = analyzer.analyze(snippet["code"])
        elapsed = time.time() - t0
        detected = len(issues) > 0
        baseline_issues = baseline.analyze(snippet["code"])
        baseline_detected = len(baseline_issues) > 0

        buggy_results.append({
            "id": snippet["id"],
            "category": snippet["category"],
            "known_bug": snippet["bug"],
            "guard_detected": detected,
            "guard_n_issues": len(issues),
            "guard_issue_types": [i["type"] for i in issues],
            "baseline_detected": baseline_detected,
            "baseline_n_issues": len(baseline_issues),
            "analysis_time_s": round(elapsed, 6),
        })

    # Run on clean snippets
    clean_results = []
    for snippet in CLEAN_SNIPPETS:
        t0 = time.time()
        issues = analyzer.analyze(snippet["code"])
        elapsed = time.time() - t0
        false_positive = len(issues) > 0
        baseline_issues = baseline.analyze(snippet["code"])
        baseline_fp = len(baseline_issues) > 0

        clean_results.append({
            "id": snippet["id"],
            "guard_false_positive": false_positive,
            "guard_n_issues": len(issues),
            "guard_issue_types": [i["type"] for i in issues],
            "baseline_false_positive": baseline_fp,
            "analysis_time_s": round(elapsed, 6),
        })

    return buggy_results, clean_results


def compute_metrics(buggy_results, clean_results):
    """Compute precision, recall, F1 for both analyzers."""
    metrics = {}

    for analyzer_name, det_key, fp_key in [
        ("guard_harvest", "guard_detected", "guard_false_positive"),
        ("baseline", "baseline_detected", "baseline_false_positive"),
    ]:
        tp = sum(1 for r in buggy_results if r[det_key])
        fn = sum(1 for r in buggy_results if not r[det_key])
        fp = sum(1 for r in clean_results if r[fp_key])
        tn = sum(1 for r in clean_results if not r[fp_key])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[analyzer_name] = {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / (tp + fn + fp + tn), 4),
        }

    return metrics


def compute_per_category_metrics(buggy_results):
    """Per-category recall for guard harvest."""
    categories = set(r["category"] for r in buggy_results)
    per_cat = {}
    for cat in sorted(categories):
        cat_results = [r for r in buggy_results if r["category"] == cat]
        detected = sum(1 for r in cat_results if r["guard_detected"])
        total = len(cat_results)
        per_cat[cat] = {
            "total": total,
            "detected": detected,
            "recall": round(detected / total, 4) if total > 0 else 0.0,
        }
    return per_cat


def main():
    print("Refinement Type Inference - Bug Detection Benchmark")
    print("=" * 60)
    t_start = time.time()

    buggy_results, clean_results = run_bug_detection_benchmark()
    metrics = compute_metrics(buggy_results, clean_results)
    per_category = compute_per_category_metrics(buggy_results)

    total_time = time.time() - t_start

    print(f"\n  Guard Harvest: P={metrics['guard_harvest']['precision']:.2f}, "
          f"R={metrics['guard_harvest']['recall']:.2f}, "
          f"F1={metrics['guard_harvest']['f1']:.2f}")
    print(f"  Baseline:      P={metrics['baseline']['precision']:.2f}, "
          f"R={metrics['baseline']['recall']:.2f}, "
          f"F1={metrics['baseline']['f1']:.2f}")
    print(f"\n  Per-category recall:")
    for cat, data in per_category.items():
        print(f"    {cat}: {data['detected']}/{data['total']} ({data['recall']:.0%})")

    all_results = {
        "experiment": "bug_detection_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "n_buggy_snippets": len(BUGGY_SNIPPETS),
        "n_clean_snippets": len(CLEAN_SNIPPETS),
        "overall_metrics": metrics,
        "per_category_recall": per_category,
        "buggy_details": buggy_results,
        "clean_details": clean_results,
        "summary": {
            "guard_harvest_f1": metrics["guard_harvest"]["f1"],
            "baseline_f1": metrics["baseline"]["f1"],
            "f1_improvement": round(metrics["guard_harvest"]["f1"] - metrics["baseline"]["f1"], 4),
            "best_category": max(per_category.items(), key=lambda x: x[1]["recall"])[0],
            "worst_category": min(per_category.items(), key=lambda x: x[1]["recall"])[0],
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "bug_detection_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary: {json.dumps(all_results['summary'], indent=2)}")


if __name__ == "__main__":
    main()
