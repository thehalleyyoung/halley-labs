"""
Anti-pattern and code smell detection for Python source code.
"""

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class PatternSeverity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    SUGGESTION = auto()


@dataclass
class PatternLocation:
    file: str = "<string>"
    line: int = 0
    col: int = 0
    end_line: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class Pattern:
    type: str
    severity: PatternSeverity
    location: PatternLocation
    explanation: str
    fix_suggestion: str = ""
    snippet: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.type} at {self.location}: {self.explanation}"


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

class _GodClassDetector(ast.NodeVisitor):
    """Detect classes with too many methods/attributes."""

    def __init__(self, filename: str, method_threshold: int = 15,
                 attr_threshold: int = 15) -> None:
        self.filename = filename
        self.method_threshold = method_threshold
        self.attr_threshold = attr_threshold
        self.patterns: List[Pattern] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods = [n for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attrs: Set[str] = set()
        for item in ast.walk(node):
            if (isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"):
                attrs.add(item.attr)

        if len(methods) > self.method_threshold:
            self.patterns.append(Pattern(
                type="god-class",
                severity=PatternSeverity.WARNING,
                location=PatternLocation(self.filename, node.lineno, node.col_offset),
                explanation=f"Class '{node.name}' has {len(methods)} methods (threshold: {self.method_threshold})",
                fix_suggestion="Split into smaller, focused classes using composition or inheritance",
            ))
        if len(attrs) > self.attr_threshold:
            self.patterns.append(Pattern(
                type="god-class",
                severity=PatternSeverity.WARNING,
                location=PatternLocation(self.filename, node.lineno, node.col_offset),
                explanation=f"Class '{node.name}' has {len(attrs)} attributes (threshold: {self.attr_threshold})",
                fix_suggestion="Group related attributes into separate data classes or value objects",
            ))
        self.generic_visit(node)


class _LongMethodDetector(ast.NodeVisitor):
    """Detect functions/methods that are too long."""

    def __init__(self, filename: str, threshold: int = 50) -> None:
        self.filename = filename
        self.threshold = threshold
        self.patterns: List[Pattern] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        end_line = getattr(node, "end_lineno", node.lineno)
        length = end_line - node.lineno + 1
        if length > self.threshold:
            self.patterns.append(Pattern(
                type="long-method",
                severity=PatternSeverity.WARNING,
                location=PatternLocation(self.filename, node.lineno, node.col_offset),
                explanation=f"Function '{node.name}' is {length} lines long (threshold: {self.threshold})",
                fix_suggestion="Extract logical blocks into helper functions",
            ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


class _FeatureEnvyDetector(ast.NodeVisitor):
    """Detect methods that use another object's data more than their own."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._current_class is None:
            self.generic_visit(node)
            return

        self_access = 0
        other_access: Counter = Counter()

        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id == "self":
                    self_access += 1
                else:
                    other_access[child.value.id] += 1

        for obj, count in other_access.items():
            if count > self_access and count >= 4:
                self.patterns.append(Pattern(
                    type="feature-envy",
                    severity=PatternSeverity.INFO,
                    location=PatternLocation(self.filename, node.lineno, node.col_offset),
                    explanation=(f"Method '{node.name}' accesses '{obj}' {count} times "
                                 f"but 'self' only {self_access} times"),
                    fix_suggestion=f"Consider moving this method to the '{obj}' class",
                ))

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


class _DataClumpDetector(ast.NodeVisitor):
    """Detect groups of parameters that always appear together."""

    def __init__(self, filename: str, threshold: int = 3) -> None:
        self.filename = filename
        self.threshold = threshold
        self.patterns: List[Pattern] = []
        self._param_groups: List[Tuple[str, int, List[str]]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params = [arg.arg for arg in node.args.args if arg.arg != "self"]
        if len(params) >= self.threshold:
            self._param_groups.append((node.name, node.lineno, params))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def finalize(self) -> None:
        # Find parameter groups shared between multiple functions
        if len(self._param_groups) < 2:
            return

        for i in range(len(self._param_groups)):
            for j in range(i + 1, len(self._param_groups)):
                name_i, line_i, params_i = self._param_groups[i]
                name_j, line_j, params_j = self._param_groups[j]
                common = set(params_i) & set(params_j)
                if len(common) >= self.threshold:
                    self.patterns.append(Pattern(
                        type="data-clump",
                        severity=PatternSeverity.INFO,
                        location=PatternLocation(self.filename, line_i, 0),
                        explanation=(f"Parameters {sorted(common)} appear together in "
                                     f"'{name_i}' and '{name_j}'"),
                        fix_suggestion="Consider grouping these parameters into a data class",
                    ))


class _MutableDefaultDetector(ast.NodeVisitor):
    """Detect mutable default arguments: def f(x=[]) or def f(x={})."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.patterns.append(Pattern(
                    type="mutable-default-argument",
                    severity=PatternSeverity.ERROR,
                    location=PatternLocation(self.filename, node.lineno, node.col_offset),
                    explanation=f"Mutable default argument in '{node.name}': {type(default).__name__}",
                    fix_suggestion="Use None as default and create the mutable object inside the function",
                ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


class _BareExceptDetector(ast.NodeVisitor):
    """Detect bare except clauses."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.patterns.append(Pattern(
                type="bare-except",
                severity=PatternSeverity.WARNING,
                location=PatternLocation(self.filename, node.lineno, node.col_offset),
                explanation="Bare except clause catches all exceptions including SystemExit and KeyboardInterrupt",
                fix_suggestion="Use 'except Exception:' to catch only standard exceptions",
            ))
        self.generic_visit(node)


class _UnsafeStringFormatDetector(ast.NodeVisitor):
    """Detect string formatting with % operator (potential injection)."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant):
            if isinstance(node.left.value, str):
                self.patterns.append(Pattern(
                    type="unsafe-string-format",
                    severity=PatternSeverity.INFO,
                    location=PatternLocation(self.filename, node.lineno, node.col_offset),
                    explanation="Using % string formatting; prefer f-strings or .format() for safety",
                    fix_suggestion="Use f-string: f\"...{variable}...\" or \"\".format(variable)",
                ))
        self.generic_visit(node)


class _BadComparisonDetector(ast.NodeVisitor):
    """Detect comparison to None/True/False with == instead of 'is'."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if isinstance(comp, ast.Constant) and comp.value in (None, True, False):
                    val_name = repr(comp.value)
                    is_op = "is" if isinstance(op, ast.Eq) else "is not"
                    self.patterns.append(Pattern(
                        type="bad-comparison",
                        severity=PatternSeverity.WARNING,
                        location=PatternLocation(self.filename, node.lineno, node.col_offset),
                        explanation=f"Comparison to {val_name} using {'==' if isinstance(op, ast.Eq) else '!='} instead of '{is_op}'",
                        fix_suggestion=f"Use '{is_op} {val_name}' instead",
                    ))
        self.generic_visit(node)


class _UnnecessaryListCompDetector(ast.NodeVisitor):
    """Detect list comprehensions that could be generators."""

    CONSUMER_FUNCS = {"sum", "min", "max", "any", "all", "sorted", "set",
                      "frozenset", "tuple", "list", "dict", "join",
                      "enumerate"}

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.patterns: List[Pattern] = []

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in self.CONSUMER_FUNCS:
            for arg in node.args:
                if isinstance(arg, ast.ListComp):
                    self.patterns.append(Pattern(
                        type="unnecessary-list-comprehension",
                        severity=PatternSeverity.INFO,
                        location=PatternLocation(self.filename, arg.lineno, arg.col_offset),
                        explanation=f"List comprehension inside {func_name}() could be a generator expression",
                        fix_suggestion=f"Replace [expr for ...] with (expr for ...) inside {func_name}()",
                    ))
        self.generic_visit(node)


class _PrimitiveObsessionDetector(ast.NodeVisitor):
    """Detect functions with too many primitive parameters."""

    def __init__(self, filename: str, threshold: int = 5) -> None:
        self.filename = filename
        self.threshold = threshold
        self.patterns: List[Pattern] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params = [a for a in node.args.args if a.arg != "self"]
        if len(params) >= self.threshold:
            self.patterns.append(Pattern(
                type="primitive-obsession",
                severity=PatternSeverity.INFO,
                location=PatternLocation(self.filename, node.lineno, node.col_offset),
                explanation=f"Function '{node.name}' has {len(params)} parameters (threshold: {self.threshold})",
                fix_suggestion="Consider grouping related parameters into a data class or named tuple",
            ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


class _ShotgunSurgeryDetector:
    """Detect functions called from many places (shotgun surgery risk)."""

    def __init__(self, filename: str, threshold: int = 5) -> None:
        self.filename = filename
        self.threshold = threshold
        self.patterns: List[Pattern] = []

    def analyze(self, tree: ast.Module) -> None:
        call_counts: Counter = Counter()
        func_locations: Dict[str, int] = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_locations[node.name] = node.lineno
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_counts[node.func.id] += 1

        for name, count in call_counts.items():
            if count >= self.threshold and name in func_locations:
                self.patterns.append(Pattern(
                    type="shotgun-surgery",
                    severity=PatternSeverity.INFO,
                    location=PatternLocation(self.filename, func_locations[name], 0),
                    explanation=f"Function '{name}' is called from {count} places; changes may have wide impact",
                    fix_suggestion="Consider the impact of changes to this function on all callers",
                ))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detect anti-patterns and code smells in Python source code."""

    def __init__(self, filename: str = "<string>",
                 method_threshold: int = 15,
                 attr_threshold: int = 15,
                 long_method_threshold: int = 50,
                 param_threshold: int = 5,
                 data_clump_threshold: int = 3,
                 shotgun_threshold: int = 5) -> None:
        self.filename = filename
        self.method_threshold = method_threshold
        self.attr_threshold = attr_threshold
        self.long_method_threshold = long_method_threshold
        self.param_threshold = param_threshold
        self.data_clump_threshold = data_clump_threshold
        self.shotgun_threshold = shotgun_threshold

    def detect(self, source_code: str) -> List[Pattern]:
        try:
            tree = ast.parse(source_code, filename=self.filename)
        except SyntaxError:
            return []

        patterns: List[Pattern] = []

        visitors: List[ast.NodeVisitor] = [
            _GodClassDetector(self.filename, self.method_threshold, self.attr_threshold),
            _LongMethodDetector(self.filename, self.long_method_threshold),
            _FeatureEnvyDetector(self.filename),
            _DataClumpDetector(self.filename, self.data_clump_threshold),
            _MutableDefaultDetector(self.filename),
            _BareExceptDetector(self.filename),
            _UnsafeStringFormatDetector(self.filename),
            _BadComparisonDetector(self.filename),
            _UnnecessaryListCompDetector(self.filename),
            _PrimitiveObsessionDetector(self.filename, self.param_threshold),
        ]

        for v in visitors:
            v.visit(tree)
            if hasattr(v, "finalize"):
                v.finalize()
            patterns.extend(v.patterns)

        # Non-visitor detectors
        shotgun = _ShotgunSurgeryDetector(self.filename, self.shotgun_threshold)
        shotgun.analyze(tree)
        patterns.extend(shotgun.patterns)

        return patterns
