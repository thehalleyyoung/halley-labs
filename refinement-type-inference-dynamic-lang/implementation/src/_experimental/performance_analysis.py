"""Static performance analysis for Python source code.

Provides AST-based detection of performance issues:
- Common performance anti-patterns
- N+1 query patterns (nested loops, repeated calls)
- Memory leak detection (reference cycles, unclosed resources)
- Caching opportunity suggestions
- Big-O complexity estimation per function
- Hotspot prediction (likely slow code paths)
"""
from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PerfSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PerfIssue:
    line: int
    message: str
    severity: PerfSeverity
    category: str
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        fix = f" -> {self.fix_suggestion}" if self.fix_suggestion else ""
        return f"line {self.line}: [{self.severity.value}] {self.message}{fix}"


@dataclass
class NPlusOne:
    outer_line: int
    inner_line: int
    outer_kind: str  # "for", "while"
    inner_pattern: str
    message: str

    def __str__(self) -> str:
        return f"line {self.outer_line}-{self.inner_line}: N+1 {self.inner_pattern}: {self.message}"


@dataclass
class MemoryLeak:
    line: int
    resource_type: str
    message: str
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        fix = f" -> {self.fix_suggestion}" if self.fix_suggestion else ""
        return f"line {self.line}: [{self.resource_type}] {self.message}{fix}"


@dataclass
class CachingSuggestion:
    line: int
    function_name: str
    reason: str
    suggestion: str

    def __str__(self) -> str:
        return f"line {self.line}: '{self.function_name}': {self.reason} -> {self.suggestion}"


@dataclass
class Hotspot:
    line: int
    function_name: str
    score: float  # 0-100, higher = more likely to be slow
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"line {self.line}: '{self.function_name}' (score: {self.score:.0f}) {', '.join(self.reasons)}"


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
        parts: List[str] = []
        obj = node.func
        while isinstance(obj, ast.Attribute):
            parts.append(obj.attr)
            obj = obj.value
        if isinstance(obj, ast.Name):
            parts.append(obj.id)
        return ".".join(reversed(parts))
    return ""


def _count_loop_depth(node: ast.AST) -> int:
    """Count the maximum nested loop depth in an AST subtree."""
    if isinstance(node, (ast.For, ast.While)):
        child_depth = max((_count_loop_depth(c) for c in ast.iter_child_nodes(node)), default=0)
        return 1 + child_depth
    return max((_count_loop_depth(c) for c in ast.iter_child_nodes(node)), default=0)


def _find_loops(node: ast.AST) -> List[ast.AST]:
    """Find all for/while loop nodes."""
    loops: List[ast.AST] = []
    for child in ast.walk(node):
        if isinstance(child, (ast.For, ast.While)):
            loops.append(child)
    return loops


# ---------------------------------------------------------------------------
# Performance issue detection
# ---------------------------------------------------------------------------

_SLOW_PATTERNS: Dict[str, Tuple[str, str, PerfSeverity]] = {
    "sorted": ("Repeated sorting", "Sort once and cache the result", PerfSeverity.MEDIUM),
    "deepcopy": ("Deep copy is expensive", "Consider shallow copy or structural sharing", PerfSeverity.MEDIUM),
    "compile": ("Regex compilation in loop", "Compile regex outside the loop", PerfSeverity.HIGH),
}

_INEFFICIENT_COLLECTION_OPS = {
    "append": "list.append in tight loop — consider list comprehension",
    "extend": "list.extend may be replaceable with concatenation",
    "insert": "list.insert(0, ...) is O(n) — consider collections.deque",
}


class _PerfVisitor(ast.NodeVisitor):
    """Detect common performance anti-patterns."""

    def __init__(self) -> None:
        self.issues: List[PerfIssue] = []
        self._in_loop = False
        self._loop_depth = 0

    def visit_For(self, node: ast.For) -> None:
        self._enter_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._enter_loop(node)

    def _enter_loop(self, node: ast.AST) -> None:
        old_in_loop = self._in_loop
        old_depth = self._loop_depth
        self._in_loop = True
        self._loop_depth += 1
        self.generic_visit(node)
        self._in_loop = old_in_loop
        self._loop_depth = old_depth

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        short_name = name.split(".")[-1] if "." in name else name

        # Slow calls in loops
        if self._in_loop and short_name in _SLOW_PATTERNS:
            msg, fix, sev = _SLOW_PATTERNS[short_name]
            self.issues.append(PerfIssue(
                line=node.lineno,
                message=f"{msg} inside loop",
                severity=sev,
                category="loop_performance",
                fix_suggestion=fix,
            ))

        # String concatenation in loops
        if self._in_loop and short_name == "join":
            pass  # join is fine
        elif self._in_loop and short_name == "format":
            pass  # format in loop is normal

        # Global imports inside functions
        if short_name == "__import__":
            self.issues.append(PerfIssue(
                line=node.lineno,
                message="Dynamic __import__ is slow",
                severity=PerfSeverity.LOW,
                category="import",
                fix_suggestion="Use static imports at module level",
            ))

        # Inefficient collection operations
        if self._in_loop and short_name == "insert":
            for a in node.args:
                if isinstance(a, ast.Constant) and a.value == 0:
                    self.issues.append(PerfIssue(
                        line=node.lineno,
                        message="list.insert(0, ...) is O(n) in a loop — O(n²) total",
                        severity=PerfSeverity.HIGH,
                        category="data_structure",
                        fix_suggestion="Use collections.deque.appendleft()",
                    ))

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # String concatenation in loops: str + str
        if self._in_loop and isinstance(node.op, ast.Add):
            if (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)) or \
               (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                self.issues.append(PerfIssue(
                    line=node.lineno,
                    message="String concatenation in loop creates many temporary strings",
                    severity=PerfSeverity.MEDIUM,
                    category="string",
                    fix_suggestion="Use list + ''.join() or io.StringIO",
                ))
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        # Nested comprehensions
        for gen in node.generators:
            if isinstance(gen.iter, ast.ListComp):
                self.issues.append(PerfIssue(
                    line=node.lineno,
                    message="Nested list comprehension — may be hard to optimize",
                    severity=PerfSeverity.LOW,
                    category="comprehension",
                    fix_suggestion="Consider itertools or explicit loops for clarity",
                ))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # `x in list_literal` in a loop — should use a set
        if self._in_loop:
            for op in node.ops:
                if isinstance(op, ast.In):
                    for comp in node.comparators:
                        if isinstance(comp, ast.List) and len(comp.elts) > 5:
                            self.issues.append(PerfIssue(
                                line=node.lineno,
                                message="'in' check against large list literal in loop — O(n) per check",
                                severity=PerfSeverity.HIGH,
                                category="data_structure",
                                fix_suggestion="Convert to a set for O(1) lookups",
                            ))
        self.generic_visit(node)


def find_performance_issues(source: str) -> List[PerfIssue]:
    """Find performance anti-patterns in Python source code.

    Detects inefficient operations in loops, poor data structure choices,
    and common performance pitfalls.
    """
    tree = _parse_source(source)
    if tree is None:
        return []
    v = _PerfVisitor()
    v.visit(tree)
    return v.issues


# ---------------------------------------------------------------------------
# N+1 detection
# ---------------------------------------------------------------------------

_DB_CALL_PATTERNS = {
    "execute", "fetchone", "fetchall", "fetchmany",
    "query", "filter", "get", "find", "find_one", "find_many",
    "select", "insert", "update", "delete",
    "cursor", "commit",
}

_NETWORK_CALL_PATTERNS = {
    "get", "post", "put", "delete", "patch", "head", "options",
    "request", "urlopen", "fetch", "send",
}


class _NPlusOneVisitor(ast.NodeVisitor):
    """Detect N+1 patterns: expensive calls inside loops."""

    def __init__(self) -> None:
        self.issues: List[NPlusOne] = []
        self._loop_stack: List[ast.AST] = []

    def visit_For(self, node: ast.For) -> None:
        self._loop_stack.append(node)
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        self._loop_stack.append(node)
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self._loop_stack:
            name = _get_call_name(node)
            short = name.split(".")[-1] if "." in name else name

            outer = self._loop_stack[-1]
            outer_kind = "for" if isinstance(outer, ast.For) else "while"

            if short in _DB_CALL_PATTERNS:
                self.issues.append(NPlusOne(
                    outer_line=outer.lineno,
                    inner_line=node.lineno,
                    outer_kind=outer_kind,
                    inner_pattern="database",
                    message=f"Database call '{name}' inside {outer_kind} loop — potential N+1 query",
                ))

            if short in _NETWORK_CALL_PATTERNS and "." in name:
                # Only flag if it looks like requests.get, not just any .get()
                prefix = name.rsplit(".", 1)[0]
                if prefix in ("requests", "httpx", "urllib", "aiohttp", "session", "client", "http"):
                    self.issues.append(NPlusOne(
                        outer_line=outer.lineno,
                        inner_line=node.lineno,
                        outer_kind=outer_kind,
                        inner_pattern="network",
                        message=f"Network call '{name}' inside {outer_kind} loop — potential N+1 request",
                    ))

            # Nested loop with inner call
            if len(self._loop_stack) >= 2:
                if short in ("append", "extend", "add"):
                    pass  # Normal
                elif short in _DB_CALL_PATTERNS or short in _NETWORK_CALL_PATTERNS:
                    pass  # Already caught above
                elif short in ("sorted", "list", "set", "dict", "tuple"):
                    self.issues.append(NPlusOne(
                        outer_line=self._loop_stack[0].lineno,
                        inner_line=node.lineno,
                        outer_kind="nested",
                        inner_pattern="collection",
                        message=f"Collection creation '{name}' in nested loop — may be O(n²+)",
                    ))

        self.generic_visit(node)


def detect_n_plus_one(source: str) -> List[NPlusOne]:
    """Detect N+1 patterns: database queries, network calls, or expensive
    operations nested inside loops.
    """
    tree = _parse_source(source)
    if tree is None:
        return []
    v = _NPlusOneVisitor()
    v.visit(tree)
    return v.issues


# ---------------------------------------------------------------------------
# Memory leak detection
# ---------------------------------------------------------------------------

_RESOURCE_TYPES = {
    "open": "file",
    "socket": "socket",
    "connect": "connection",
    "cursor": "cursor",
    "urlopen": "url_handle",
    "Popen": "subprocess",
    "TemporaryFile": "tempfile",
    "NamedTemporaryFile": "tempfile",
}


class _MemoryLeakVisitor(ast.NodeVisitor):
    """Detect potential memory leaks and resource leaks."""

    def __init__(self) -> None:
        self.issues: List[MemoryLeak] = []
        self._in_with = False

    def visit_With(self, node: ast.With) -> None:
        old = self._in_with
        self._in_with = True
        self.generic_visit(node)
        self._in_with = old

    visit_AsyncWith = visit_With

    def visit_Assign(self, node: ast.Assign) -> None:
        # Check for resource assignment without context manager
        if not self._in_with and isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            short = name.split(".")[-1] if "." in name else name
            if short in _RESOURCE_TYPES:
                self.issues.append(MemoryLeak(
                    line=node.lineno,
                    resource_type=_RESOURCE_TYPES[short],
                    message=f"'{name}' opened without context manager — may leak",
                    fix_suggestion=f"Use 'with {name}(...) as var:' instead",
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)

        # Circular reference patterns
        if name == "setattr":
            if len(node.args) >= 3:
                # setattr(obj, 'parent', self) — potential circular ref
                if isinstance(node.args[2], ast.Name) and node.args[2].id == "self":
                    self.issues.append(MemoryLeak(
                        line=node.lineno,
                        resource_type="circular_ref",
                        message="setattr with self reference may create circular reference",
                        fix_suggestion="Use weakref.ref() to break the cycle",
                    ))

        # Large data accumulation patterns
        if name in ("append", "extend", "add", "update"):
            # Check if target is a module-level or class-level variable
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # Could be accumulating in a global list
                pass  # Heuristic only, flag in specific contexts

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for __del__ methods that might prevent GC
        if node.name == "__del__":
            self.issues.append(MemoryLeak(
                line=node.lineno,
                resource_type="finalizer",
                message="__del__ method may prevent garbage collection of cycles",
                fix_suggestion="Use weakref.finalize() or context managers instead",
            ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.issues.append(MemoryLeak(
                line=node.lineno,
                resource_type="global_state",
                message=f"Global variable '{name}' may accumulate data over time",
                fix_suggestion="Consider scoping or periodic cleanup",
            ))
        self.generic_visit(node)


def detect_memory_leaks(source: str) -> List[MemoryLeak]:
    """Detect potential memory and resource leaks in Python source code.

    Finds:
    - Resources opened without context managers
    - Circular reference patterns
    - __del__ methods that may prevent GC
    - Global state accumulation
    """
    tree = _parse_source(source)
    if tree is None:
        return []
    v = _MemoryLeakVisitor()
    v.visit(tree)
    return v.issues


# ---------------------------------------------------------------------------
# Caching suggestions
# ---------------------------------------------------------------------------

class _CachingVisitor(ast.NodeVisitor):
    """Identify functions that could benefit from caching."""

    def __init__(self) -> None:
        self.suggestions: List[CachingSuggestion] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_cacheable(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_cacheable(self, node: ast.FunctionDef) -> None:
        # Skip if already cached
        for dec in node.decorator_list:
            dec_name = ""
            if isinstance(dec, ast.Name):
                dec_name = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_name = dec.attr
            elif isinstance(dec, ast.Call):
                dec_name = _get_call_name(dec)
            if dec_name in ("cache", "lru_cache", "cached_property", "memoize"):
                return

        # Check if function is pure (no side effects, no I/O)
        has_side_effects = False
        has_loop = False
        call_count = 0
        is_recursive = False

        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                has_loop = True
            if isinstance(child, ast.Call):
                call_count += 1
                name = _get_call_name(child)
                short = name.split(".")[-1] if "." in name else name
                if short in ("print", "write", "send", "execute", "commit", "save"):
                    has_side_effects = True
                if short == node.name:
                    is_recursive = True
            if isinstance(child, ast.Global):
                has_side_effects = True

        if has_side_effects:
            return

        # Recursive function without memoization
        if is_recursive:
            self.suggestions.append(CachingSuggestion(
                line=node.lineno,
                function_name=node.name,
                reason="Recursive function without memoization",
                suggestion="Add @functools.lru_cache() decorator",
            ))
            return

        # Pure function with expensive computation (has loops or many calls)
        if not has_side_effects and (has_loop or call_count > 3):
            # Only suggest if function takes hashable args
            args = node.args.args
            param_names = [a.arg for a in args if a.arg not in ("self", "cls")]
            if param_names:
                self.suggestions.append(CachingSuggestion(
                    line=node.lineno,
                    function_name=node.name,
                    reason="Pure function with expensive computation",
                    suggestion="Consider @functools.lru_cache() if args are hashable",
                ))


def suggest_caching(source: str) -> List[CachingSuggestion]:
    """Suggest functions that could benefit from caching/memoization.

    Identifies recursive functions without memoization and pure functions
    with expensive computation patterns.
    """
    tree = _parse_source(source)
    if tree is None:
        return []
    v = _CachingVisitor()
    v.visit(tree)
    return v.suggestions


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

def _estimate_complexity(node: ast.FunctionDef) -> str:
    """Estimate the Big-O time complexity of a function."""
    max_loop_depth = 0
    has_sort = False
    has_recursion = False
    has_binary_search_pattern = False

    for child in ast.walk(node):
        if isinstance(child, (ast.For, ast.While)):
            depth = _count_loop_depth(child)
            if depth > max_loop_depth:
                max_loop_depth = depth

        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            short = name.split(".")[-1] if "." in name else name
            if short in ("sort", "sorted"):
                has_sort = True
            if short == node.name:
                has_recursion = True
            if short in ("bisect", "bisect_left", "bisect_right"):
                has_binary_search_pattern = True

    # Determine complexity
    if max_loop_depth == 0 and not has_sort and not has_recursion:
        return "O(1)"

    if has_binary_search_pattern:
        return "O(log n)"

    if has_recursion:
        if max_loop_depth == 0:
            # Simple recursion without loops — could be O(n) or O(2^n)
            # Check for binary tree pattern (two recursive calls)
            rec_count = sum(
                1 for c in ast.walk(node)
                if isinstance(c, ast.Call) and _get_call_name(c) == node.name
            )
            if rec_count >= 2:
                return "O(2^n)"
            return "O(n)"
        return f"O(n^{max_loop_depth + 1})"

    if has_sort and max_loop_depth == 0:
        return "O(n log n)"

    if has_sort and max_loop_depth >= 1:
        return f"O(n^{max_loop_depth} * n log n)"

    if max_loop_depth == 1:
        return "O(n)"
    if max_loop_depth == 2:
        return "O(n²)"
    if max_loop_depth == 3:
        return "O(n³)"
    return f"O(n^{max_loop_depth})"


def complexity_estimation(source: str) -> Dict[str, str]:
    """Estimate Big-O time complexity for each function in the source.

    Uses heuristics based on loop nesting depth, sorting calls,
    and recursion patterns.
    """
    tree = _parse_source(source)
    if tree is None:
        return {}

    result: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = _estimate_complexity(node)
    return result


# ---------------------------------------------------------------------------
# Hotspot prediction
# ---------------------------------------------------------------------------

def hotspot_prediction(source: str) -> List[Hotspot]:
    """Predict which functions are likely performance hotspots.

    Scores each function based on:
    - Loop nesting depth
    - Number of function calls
    - Presence of I/O operations
    - String operations in loops
    - Data structure usage patterns
    """
    tree = _parse_source(source)
    if tree is None:
        return []

    hotspots: List[Hotspot] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        score = 0.0
        reasons: List[str] = []

        # Loop depth
        loop_depth = _count_loop_depth(node)
        if loop_depth >= 3:
            score += 40
            reasons.append(f"deep nesting ({loop_depth} levels)")
        elif loop_depth == 2:
            score += 25
            reasons.append("nested loops")
        elif loop_depth == 1:
            score += 10

        # Call count
        call_count = sum(1 for c in ast.walk(node) if isinstance(c, ast.Call))
        if call_count > 20:
            score += 15
            reasons.append(f"many calls ({call_count})")
        elif call_count > 10:
            score += 8

        # I/O operations
        io_calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _get_call_name(child)
                short = name.split(".")[-1] if "." in name else name
                if short in ("read", "write", "open", "execute", "fetch", "send", "recv", "connect"):
                    io_calls.add(short)
        if io_calls:
            score += len(io_calls) * 10
            reasons.append(f"I/O ops: {', '.join(io_calls)}")

        # Function length
        func_lines = 0
        if hasattr(node, "end_lineno") and node.end_lineno:
            func_lines = node.end_lineno - node.lineno
        if func_lines > 100:
            score += 15
            reasons.append(f"long function ({func_lines} lines)")
        elif func_lines > 50:
            score += 8

        # Recursion
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if _get_call_name(child) == node.name:
                    score += 20
                    reasons.append("recursive")
                    break

        # Only report functions with notable score
        if score >= 15 and reasons:
            hotspots.append(Hotspot(
                line=node.lineno,
                function_name=node.name,
                score=min(score, 100),
                reasons=reasons,
            ))

    return sorted(hotspots, key=lambda h: h.score, reverse=True)
