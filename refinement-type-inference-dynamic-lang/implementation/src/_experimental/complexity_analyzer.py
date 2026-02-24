"""
Code complexity measurement: cyclomatic, cognitive, Halstead,
maintainability index, ABC, LCOM, CBO, nesting depth, fan-in/fan-out.
"""

import ast
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FunctionComplexity:
    name: str
    lineno: int = 0
    end_lineno: int = 0
    cyclomatic: int = 1
    cognitive: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_estimated_bugs: float = 0.0
    maintainability_index: float = 100.0
    abc_score: float = 0.0
    abc_assignments: int = 0
    abc_branches: int = 0
    abc_conditions: int = 0
    nesting_depth: int = 0
    length: int = 0
    parameter_count: int = 0


@dataclass
class ClassComplexity:
    name: str
    lineno: int = 0
    method_count: int = 0
    attribute_count: int = 0
    lcom: float = 0.0
    weighted_methods: float = 0.0
    max_method_complexity: float = 0.0
    cbo: int = 0


@dataclass
class ModuleComplexity:
    filename: str = "<string>"
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    fan_in: int = 0
    fan_out: int = 0
    imports: List[str] = field(default_factory=list)


@dataclass
class Hotspot:
    name: str
    file: str
    line: int
    metric: str
    value: float
    threshold: float
    recommendation: str


@dataclass
class ComplexityReport:
    per_function: List[FunctionComplexity] = field(default_factory=list)
    per_class: List[ClassComplexity] = field(default_factory=list)
    per_module: List[ModuleComplexity] = field(default_factory=list)
    hotspots: List[Hotspot] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def avg_cyclomatic(self) -> float:
        if not self.per_function:
            return 0.0
        return sum(f.cyclomatic for f in self.per_function) / len(self.per_function)

    @property
    def max_cyclomatic(self) -> float:
        if not self.per_function:
            return 0.0
        return max(f.cyclomatic for f in self.per_function)

    @property
    def total_cyclomatic(self) -> int:
        return sum(f.cyclomatic for f in self.per_function)


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------

class _CyclomaticVisitor(ast.NodeVisitor):
    """Count decision points to compute cyclomatic complexity."""

    def __init__(self) -> None:
        self.complexity = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each `and`/`or` adds a decision path
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1
        self.complexity += len(node.ifs)
        self.generic_visit(node)


def cyclomatic_complexity(node: ast.AST) -> int:
    v = _CyclomaticVisitor()
    v.visit(node)
    return v.complexity


# ---------------------------------------------------------------------------
# Cognitive complexity (Sonar-style)
# ---------------------------------------------------------------------------

class _CognitiveVisitor(ast.NodeVisitor):
    """Compute cognitive complexity with nesting increments."""

    def __init__(self) -> None:
        self.score = 0
        self._nesting = 0

    def _increment(self, base: int = 1) -> None:
        self.score += base + self._nesting

    def visit_If(self, node: ast.If) -> None:
        self._increment()
        self._nesting += 1
        for stmt in node.body:
            self.visit(stmt)
        self._nesting -= 1
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif: increment but don't nest
                self.score += 1
                self.visit(node.orelse[0])
            else:
                self.score += 1  # else
                self._nesting += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self._nesting -= 1

    def visit_For(self, node: ast.For) -> None:
        self._increment()
        self._nesting += 1
        for stmt in node.body:
            self.visit(stmt)
        self._nesting -= 1
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._increment()
        self._nesting += 1
        for stmt in node.body:
            self.visit(stmt)
        self._nesting -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self._increment()
        self._nesting += 1
        for stmt in node.body:
            self.visit(stmt)
        self._nesting -= 1
        for handler in node.handlers:
            self._increment()
            self._nesting += 1
            for stmt in handler.body:
                self.visit(stmt)
            self._nesting -= 1

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._increment()
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        self.score += 1

    def visit_Continue(self, node: ast.Continue) -> None:
        self.score += 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested function
        self._nesting += 1
        for stmt in node.body:
            self.visit(stmt)
        self._nesting -= 1

    visit_AsyncFunctionDef = visit_FunctionDef


def cognitive_complexity(node: ast.AST) -> int:
    v = _CognitiveVisitor()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for stmt in node.body:
            v.visit(stmt)
    else:
        v.visit(node)
    return v.score


# ---------------------------------------------------------------------------
# Halstead metrics
# ---------------------------------------------------------------------------

class _HalsteadCollector(ast.NodeVisitor):
    """Collect operators and operands from AST."""

    def __init__(self) -> None:
        self.operators: List[str] = []
        self.operands: List[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.operators.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            self.operators.append(type(op).__name__)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.operators.append(type(node.op).__name__ + "=")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.operators.append("=")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.operators.append("()")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.operators.append("[]")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.operators.append(".")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.operators.append("return")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.operators.append("if")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.operators.append("for")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.operators.append("while")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.operands.append(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.operands.append(repr(node.value))


@dataclass
class HalsteadMetrics:
    n1: int = 0   # unique operators
    n2: int = 0   # unique operands
    N1: int = 0   # total operators
    N2: int = 0   # total operands
    vocabulary: int = 0
    length: int = 0
    volume: float = 0.0
    difficulty: float = 0.0
    effort: float = 0.0
    estimated_bugs: float = 0.0
    time_to_program: float = 0.0


def halstead_metrics(node: ast.AST) -> HalsteadMetrics:
    c = _HalsteadCollector()
    c.visit(node)

    n1 = len(set(c.operators))
    n2 = len(set(c.operands))
    N1 = len(c.operators)
    N2 = len(c.operands)

    vocabulary = n1 + n2
    length = N1 + N2

    if vocabulary == 0:
        return HalsteadMetrics(n1=n1, n2=n2, N1=N1, N2=N2)

    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0.0
    difficulty = (n1 / 2.0) * (N2 / n2) if n2 > 0 else 0.0
    effort = difficulty * volume
    estimated_bugs = volume / 3000.0
    time_to_program = effort / 18.0

    return HalsteadMetrics(
        n1=n1, n2=n2, N1=N1, N2=N2,
        vocabulary=vocabulary, length=length,
        volume=volume, difficulty=difficulty,
        effort=effort, estimated_bugs=estimated_bugs,
        time_to_program=time_to_program,
    )


# ---------------------------------------------------------------------------
# ABC metric
# ---------------------------------------------------------------------------

class _ABCVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignments = 0
        self.branches = 0
        self.conditions = 0

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assignments += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.assignments += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.conditions += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.conditions += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.conditions += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.conditions += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.conditions += len(node.values) - 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.conditions += len(node.ops)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.conditions += 1
        self.generic_visit(node)

    @property
    def score(self) -> float:
        return math.sqrt(
            self.assignments ** 2 + self.branches ** 2 + self.conditions ** 2
        )


def abc_metric(node: ast.AST) -> Tuple[int, int, int, float]:
    v = _ABCVisitor()
    v.visit(node)
    return v.assignments, v.branches, v.conditions, v.score


# ---------------------------------------------------------------------------
# Nesting depth
# ---------------------------------------------------------------------------

class _NestingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.max_depth = 0
        self._current = 0

    def _enter(self) -> None:
        self._current += 1
        self.max_depth = max(self.max_depth, self._current)

    def _leave(self) -> None:
        self._current -= 1

    def visit_If(self, node: ast.If) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    def visit_For(self, node: ast.For) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    def visit_With(self, node: ast.With) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    visit_AsyncWith = visit_With

    def visit_Try(self, node: ast.Try) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter()
        self.generic_visit(node)
        self._leave()

    visit_AsyncFunctionDef = visit_FunctionDef


def max_nesting_depth(node: ast.AST) -> int:
    v = _NestingVisitor()
    v.visit(node)
    return v.max_depth


# ---------------------------------------------------------------------------
# LCOM (Lack of Cohesion of Methods)
# ---------------------------------------------------------------------------

def lcom(cls_node: ast.ClassDef) -> float:
    """LCOM4: number of connected components in the method-attribute graph."""
    methods: Dict[str, Set[str]] = {}
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs: Set[str] = set()
            for node in ast.walk(item):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"):
                    attrs.add(node.attr)
            methods[item.name] = attrs

    if not methods:
        return 0.0

    # Build adjacency: two methods are connected if they share an attribute
    method_names = list(methods.keys())
    parent: Dict[str, str] = {m: m for m in method_names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            if methods[method_names[i]] & methods[method_names[j]]:
                union(method_names[i], method_names[j])

    components = len(set(find(m) for m in method_names))
    return float(components)


# ---------------------------------------------------------------------------
# CBO (Coupling Between Objects)
# ---------------------------------------------------------------------------

def cbo(cls_node: ast.ClassDef, all_class_names: Set[str]) -> int:
    """Count distinct external classes referenced."""
    referenced: Set[str] = set()
    for node in ast.walk(cls_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in all_class_names and node.id != cls_node.name:
                referenced.add(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in all_class_names and node.value.id != cls_node.name:
                referenced.add(node.value.id)
    return len(referenced)


# ---------------------------------------------------------------------------
# Maintainability index
# ---------------------------------------------------------------------------

def maintainability_index(volume: float, cyclo: int, loc: int) -> float:
    """Microsoft-style maintainability index (0-100 scale)."""
    if loc == 0:
        return 100.0
    ln_vol = math.log(volume) if volume > 0 else 0.0
    ln_loc = math.log(loc) if loc > 0 else 0.0
    mi = 171.0 - 5.2 * ln_vol - 0.23 * cyclo - 16.2 * ln_loc
    return max(0.0, min(100.0, mi * 100.0 / 171.0))


# ---------------------------------------------------------------------------
# Module-level metrics
# ---------------------------------------------------------------------------

def _count_lines(source: str) -> Tuple[int, int, int, int]:
    lines = source.split("\n")
    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comment = sum(1 for l in lines if l.strip().startswith("#"))
    code = total - blank - comment
    return total, code, comment, blank


def _collect_imports(tree: ast.Module) -> List[str]:
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}")
    return imports


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """Analyse Python source code for various complexity metrics."""

    CYCLOMATIC_THRESHOLD = 10
    COGNITIVE_THRESHOLD = 15
    FUNCTION_LENGTH_THRESHOLD = 50
    NESTING_THRESHOLD = 4
    LCOM_THRESHOLD = 3
    METHOD_COUNT_THRESHOLD = 20
    ABC_THRESHOLD = 30.0
    MI_THRESHOLD = 40.0

    def __init__(self, filename: str = "<string>") -> None:
        self.filename = filename

    def analyze(self, source_code: str) -> ComplexityReport:
        try:
            tree = ast.parse(source_code, filename=self.filename)
        except SyntaxError:
            return ComplexityReport()

        report = ComplexityReport()

        # Module-level
        total, code, comment, blank = _count_lines(source_code)
        imports = _collect_imports(tree)
        mod = ModuleComplexity(
            filename=self.filename,
            total_lines=total,
            code_lines=code,
            comment_lines=comment,
            blank_lines=blank,
            fan_out=len(set(imports)),
            imports=imports,
        )
        report.per_module.append(mod)

        # Collect all class names for CBO
        all_class_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                all_class_names.add(node.name)

        # Functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fc = self._analyze_function(node, source_code)
                report.per_function.append(fc)

        # Classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cc = self._analyze_class(node, all_class_names)
                report.per_class.append(cc)

        # Hotspots & recommendations
        self._find_hotspots(report)

        return report

    def _analyze_function(self, node: ast.FunctionDef, source: str) -> FunctionComplexity:
        end_line = getattr(node, "end_lineno", node.lineno)
        length = end_line - node.lineno + 1

        cc = cyclomatic_complexity(node)
        cog = cognitive_complexity(node)
        h = halstead_metrics(node)
        a, b, c, abc_score = abc_metric(node)
        depth = max_nesting_depth(node)
        mi = maintainability_index(h.volume, cc, length)

        param_count = len(node.args.args)
        if node.args.vararg:
            param_count += 1
        if node.args.kwarg:
            param_count += 1

        return FunctionComplexity(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_line,
            cyclomatic=cc,
            cognitive=cog,
            halstead_volume=h.volume,
            halstead_difficulty=h.difficulty,
            halstead_effort=h.effort,
            halstead_estimated_bugs=h.estimated_bugs,
            maintainability_index=mi,
            abc_score=abc_score,
            abc_assignments=a,
            abc_branches=b,
            abc_conditions=c,
            nesting_depth=depth,
            length=length,
            parameter_count=param_count,
        )

    def _analyze_class(self, node: ast.ClassDef,
                       all_class_names: Set[str]) -> ClassComplexity:
        methods = [n for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attrs: Set[str] = set()
        for item in ast.walk(node):
            if (isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"):
                attrs.add(item.attr)

        method_complexities = [cyclomatic_complexity(m) for m in methods]
        wmc = sum(method_complexities) if method_complexities else 0

        return ClassComplexity(
            name=node.name,
            lineno=node.lineno,
            method_count=len(methods),
            attribute_count=len(attrs),
            lcom=lcom(node),
            weighted_methods=wmc,
            max_method_complexity=max(method_complexities) if method_complexities else 0,
            cbo=cbo(node, all_class_names),
        )

    def _find_hotspots(self, report: ComplexityReport) -> None:
        for fc in report.per_function:
            if fc.cyclomatic > self.CYCLOMATIC_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=fc.name, file=self.filename, line=fc.lineno,
                    metric="cyclomatic", value=fc.cyclomatic,
                    threshold=self.CYCLOMATIC_THRESHOLD,
                    recommendation=f"Consider splitting '{fc.name}' into smaller functions",
                ))
            if fc.cognitive > self.COGNITIVE_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=fc.name, file=self.filename, line=fc.lineno,
                    metric="cognitive", value=fc.cognitive,
                    threshold=self.COGNITIVE_THRESHOLD,
                    recommendation=f"Reduce cognitive complexity of '{fc.name}' by extracting conditions",
                ))
            if fc.length > self.FUNCTION_LENGTH_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=fc.name, file=self.filename, line=fc.lineno,
                    metric="length", value=fc.length,
                    threshold=self.FUNCTION_LENGTH_THRESHOLD,
                    recommendation=f"Function '{fc.name}' is too long ({fc.length} lines)",
                ))
            if fc.nesting_depth > self.NESTING_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=fc.name, file=self.filename, line=fc.lineno,
                    metric="nesting", value=fc.nesting_depth,
                    threshold=self.NESTING_THRESHOLD,
                    recommendation=f"Reduce nesting depth in '{fc.name}' using early returns",
                ))
            if fc.maintainability_index < self.MI_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=fc.name, file=self.filename, line=fc.lineno,
                    metric="maintainability_index", value=fc.maintainability_index,
                    threshold=self.MI_THRESHOLD,
                    recommendation=f"Low maintainability index for '{fc.name}': consider refactoring",
                ))

        for cc in report.per_class:
            if cc.lcom > self.LCOM_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=cc.name, file=self.filename, line=cc.lineno,
                    metric="LCOM", value=cc.lcom,
                    threshold=self.LCOM_THRESHOLD,
                    recommendation=f"Class '{cc.name}' has low cohesion; consider splitting",
                ))
            if cc.method_count > self.METHOD_COUNT_THRESHOLD:
                report.hotspots.append(Hotspot(
                    name=cc.name, file=self.filename, line=cc.lineno,
                    metric="method_count", value=cc.method_count,
                    threshold=self.METHOD_COUNT_THRESHOLD,
                    recommendation=f"Class '{cc.name}' has too many methods ({cc.method_count})",
                ))

        if report.hotspots:
            report.recommendations.append(
                f"Found {len(report.hotspots)} complexity hotspot(s) requiring attention"
            )
