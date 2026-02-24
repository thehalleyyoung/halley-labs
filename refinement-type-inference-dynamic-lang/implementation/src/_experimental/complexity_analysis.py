"""Code complexity analysis for Python source code.

Provides AST-based complexity metrics:
- Cyclomatic complexity (McCabe) per function
- Cognitive complexity per function
- Detection of deeply nested blocks
- Detection of overly long functions
- Detection of god classes (too many methods/attributes)
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class DeepNesting:
    function: str
    line: int
    depth: int
    context: str  # e.g. "if > for > if > try"

    def __str__(self) -> str:
        return f"{self.line}: nesting depth {self.depth} in '{self.function}' ({self.context})"


@dataclass
class LongFunction:
    name: str
    line: int
    length: int
    threshold: int

    def __str__(self) -> str:
        return f"{self.line}: function '{self.name}' is {self.length} lines (max {self.threshold})"


@dataclass
class GodClass:
    name: str
    line: int
    method_count: int
    attribute_count: int
    threshold: int
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.line}: class '{self.name}' has {self.method_count} methods, {self.attribute_count} attrs"


# ---------------------------------------------------------------------------
# Cyclomatic complexity (McCabe)
# ---------------------------------------------------------------------------

class _CyclomaticVisitor(ast.NodeVisitor):
    """Count decision points to compute McCabe cyclomatic complexity."""

    def __init__(self) -> None:
        self.functions: Dict[str, int] = {}
        self._current: Optional[str] = None
        self._complexity: int = 0
        self._scope_stack: List[str] = []

    def _qualified_name(self, name: str) -> str:
        if self._scope_stack:
            return ".".join(self._scope_stack) + "." + name
        return name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _enter_function(self, node: ast.FunctionDef) -> None:
        outer, outer_complexity = self._current, self._complexity
        qname = self._qualified_name(node.name)
        self._current = qname
        self._complexity = 1  # base complexity
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()
        self.functions[qname] = self._complexity
        self._current, self._complexity = outer, outer_complexity

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    # Each branch/decision adds 1
    def visit_If(self, node: ast.If) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        # with statements don't add complexity by default
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if self._current:
            # Each 'and'/'or' adds 1 for each additional operand beyond the first
            self._complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self._current:
            self._complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if self._current:
            self._complexity += 1
            self._complexity += len(node.ifs)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Cognitive complexity
# ---------------------------------------------------------------------------

class _CognitiveVisitor(ast.NodeVisitor):
    """Compute cognitive complexity per function.

    Rules: +1 for each control flow break; nesting increments add to
    the penalty for deeply nested structures.
    """

    def __init__(self) -> None:
        self.functions: Dict[str, int] = {}
        self._current: Optional[str] = None
        self._score: int = 0
        self._nesting: int = 0
        self._scope_stack: List[str] = []

    def _qualified_name(self, name: str) -> str:
        if self._scope_stack:
            return ".".join(self._scope_stack) + "." + name
        return name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _enter_function(self, node: ast.FunctionDef) -> None:
        outer_current, outer_score, outer_nesting = self._current, self._score, self._nesting
        qname = self._qualified_name(node.name)
        self._current = qname
        self._score = 0
        self._nesting = 0
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()
        self.functions[qname] = self._score
        self._current, self._score, self._nesting = outer_current, outer_score, outer_nesting

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _increment(self, nesting_penalty: bool = True) -> None:
        if self._current is None:
            return
        self._score += 1
        if nesting_penalty:
            self._score += self._nesting

    def visit_If(self, node: ast.If) -> None:
        if self._current:
            self._increment()
            self._nesting += 1
            for child in node.body:
                self.visit(child)
            self._nesting -= 1
            # 'elif' chains don't add nesting
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # elif — +1 but no nesting increase
                    self._score += 1
                    elif_node = node.orelse[0]
                    self._nesting += 1
                    for child in elif_node.body:
                        self.visit(child)
                    self._nesting -= 1
                    if elif_node.orelse:
                        for child in elif_node.orelse:
                            self.visit(child)
                else:
                    # else
                    self._score += 1
                    for child in node.orelse:
                        self.visit(child)
        else:
            self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(self, node: ast.AST) -> None:
        if self._current:
            self._increment()
            self._nesting += 1
            self.generic_visit(node)
            self._nesting -= 1
        else:
            self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._current:
            self._increment()
            self._nesting += 1
            self.generic_visit(node)
            self._nesting -= 1
        else:
            self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if self._current:
            self._score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if self._current:
            self._increment()
            self._nesting += 1
            self.generic_visit(node)
            self._nesting -= 1
        else:
            self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        if self._current:
            self._score += 1

    def visit_Continue(self, node: ast.Continue) -> None:
        if self._current:
            self._score += 1


# ---------------------------------------------------------------------------
# Nesting depth
# ---------------------------------------------------------------------------

_NESTING_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
    ast.Try, ast.ExceptHandler,
)

_NESTING_LABELS = {
    ast.If: "if", ast.For: "for", ast.AsyncFor: "async for",
    ast.While: "while", ast.With: "with", ast.AsyncWith: "async with",
    ast.Try: "try", ast.ExceptHandler: "except",
}


class _NestingVisitor(ast.NodeVisitor):

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        self.results: List[DeepNesting] = []
        self._depth: int = 0
        self._context: List[str] = []
        self._function: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self._function
        self._function = node.name
        self.generic_visit(node)
        self._function = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old = self._function
        self._function = node.name
        self.generic_visit(node)
        self._function = old

    def _visit_nesting(self, node: ast.AST) -> None:
        label = _NESTING_LABELS.get(type(node), type(node).__name__.lower())
        self._depth += 1
        self._context.append(label)
        if self._depth > self.max_depth:
            self.results.append(DeepNesting(
                function=self._function,
                line=node.lineno,
                depth=self._depth,
                context=" > ".join(self._context),
            ))
        self.generic_visit(node)
        self._context.pop()
        self._depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._visit_nesting(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_nesting(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._visit_nesting(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_nesting(node)

    visit_AsyncWith = visit_With

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_nesting(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._visit_nesting(node)


# ---------------------------------------------------------------------------
# Long functions
# ---------------------------------------------------------------------------

def _function_length(node: ast.FunctionDef) -> int:
    end = node.end_lineno or node.lineno
    return end - node.lineno + 1


# ---------------------------------------------------------------------------
# God classes
# ---------------------------------------------------------------------------

class _ClassAnalyzer(ast.NodeVisitor):

    def __init__(self, max_methods: int) -> None:
        self.max_methods = max_methods
        self.results: List[GodClass] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods: List[str] = []
        attrs: set[str] = set()

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
                # Collect self.X attributes from __init__
                if item.name == "__init__":
                    for child in ast.walk(item):
                        if (isinstance(child, ast.Assign)):
                            for target in child.targets:
                                if (isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"):
                                    attrs.add(target.attr)
                        elif isinstance(child, ast.AnnAssign):
                            if (isinstance(child.target, ast.Attribute)
                                    and isinstance(child.target.value, ast.Name)
                                    and child.target.value.id == "self"):
                                attrs.add(child.target.attr)

        reasons: List[str] = []
        if len(methods) > self.max_methods:
            reasons.append(f"{len(methods)} methods (max {self.max_methods})")
        if len(attrs) > self.max_methods:
            reasons.append(f"{len(attrs)} instance attributes (max {self.max_methods})")

        if reasons:
            self.results.append(GodClass(
                name=node.name,
                line=node.lineno,
                method_count=len(methods),
                attribute_count=len(attrs),
                threshold=self.max_methods,
                reasons=reasons,
            ))

        # Don't recurse into nested classes
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                self.visit(item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse(source: str) -> ast.Module:
    return ast.parse(source, type_comments=True)


def cyclomatic_complexity(source: str) -> Dict[str, int]:
    """Return McCabe cyclomatic complexity per function."""
    tree = _parse(source)
    visitor = _CyclomaticVisitor()
    visitor.visit(tree)
    return visitor.functions


def cognitive_complexity(source: str) -> Dict[str, int]:
    """Return cognitive complexity per function."""
    tree = _parse(source)
    visitor = _CognitiveVisitor()
    visitor.visit(tree)
    return visitor.functions


def find_deeply_nested(source: str, max_depth: int = 4) -> List[DeepNesting]:
    """Find code blocks nested deeper than *max_depth*."""
    tree = _parse(source)
    visitor = _NestingVisitor(max_depth)
    visitor.visit(tree)
    return visitor.results


def find_long_functions(source: str, max_lines: int = 50) -> List[LongFunction]:
    """Find functions longer than *max_lines*."""
    tree = _parse(source)
    results: List[LongFunction] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = _function_length(node)
            if length > max_lines:
                results.append(LongFunction(node.name, node.lineno, length, max_lines))
    return results


def find_god_classes(source: str, max_methods: int = 20) -> List[GodClass]:
    """Find classes with too many methods or attributes."""
    tree = _parse(source)
    analyzer = _ClassAnalyzer(max_methods)
    analyzer.visit(tree)
    return analyzer.results
