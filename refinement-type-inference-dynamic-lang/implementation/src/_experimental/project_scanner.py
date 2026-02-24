"""
Scan entire Python projects: aggregate bugs, complexity, dependencies,
circular imports, Python 2/3 compatibility, and health score.
"""

import ast
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from .python_ast_analyzer import PythonASTAnalyzer, AnalysisResult, Bug, Severity
except ImportError:
    try:
        from python_ast_analyzer import PythonASTAnalyzer, AnalysisResult, Bug, Severity
    except ImportError:
        PythonASTAnalyzer = None  # type: ignore[misc,assignment]
        AnalysisResult = None  # type: ignore[misc,assignment]

try:
    from .complexity_analyzer import ComplexityAnalyzer, ComplexityReport
except ImportError:
    try:
        from complexity_analyzer import ComplexityAnalyzer, ComplexityReport
    except ImportError:
        ComplexityAnalyzer = None  # type: ignore[misc,assignment]
        ComplexityReport = None  # type: ignore[misc,assignment]

try:
    from .pattern_detector import PatternDetector, Pattern
except ImportError:
    try:
        from pattern_detector import PatternDetector, Pattern
    except ImportError:
        PatternDetector = None  # type: ignore[misc,assignment]
        Pattern = None  # type: ignore[misc,assignment]

try:
    from .taint_tracker import TaintTracker, TaintFlow
except ImportError:
    try:
        from taint_tracker import TaintTracker, TaintFlow
    except ImportError:
        TaintTracker = None  # type: ignore[misc,assignment]
        TaintFlow = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    path: str
    bugs: List[Bug] = field(default_factory=list)
    patterns: List[Pattern] = field(default_factory=list)
    taint_flows: List[TaintFlow] = field(default_factory=list)
    complexity: Optional[ComplexityReport] = None
    lines: int = 0
    imports: List[str] = field(default_factory=list)


@dataclass
class DependencyEdge:
    source: str
    target: str
    import_line: int = 0


@dataclass
class ProjectReport:
    files_analyzed: int = 0
    total_bugs: int = 0
    bugs_by_severity: Dict[str, int] = field(default_factory=lambda: {
        "ERROR": 0, "WARNING": 0, "INFO": 0
    })
    bugs_by_category: Dict[str, int] = field(default_factory=dict)
    bugs_by_file: Dict[str, int] = field(default_factory=dict)
    complexity_scores: Dict[str, float] = field(default_factory=dict)
    dependency_graph: List[DependencyEdge] = field(default_factory=list)
    circular_imports: List[List[str]] = field(default_factory=list)
    health_score: float = 100.0
    recommendations: List[str] = field(default_factory=list)
    file_reports: Dict[str, FileReport] = field(default_factory=dict)
    py2_incompatibilities: List[str] = field(default_factory=list)
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    taint_flows: int = 0


# ---------------------------------------------------------------------------
# Gitignore parser (simplified)
# ---------------------------------------------------------------------------

class _GitignoreFilter:
    """Simple .gitignore-style path filter."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.patterns: List[str] = []
        gitignore = os.path.join(root, ".gitignore")
        if os.path.isfile(gitignore):
            with open(gitignore, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.patterns.append(line)

        # Always ignore common directories
        self.always_ignore = {
            "__pycache__", ".git", ".svn", ".hg", "node_modules",
            ".tox", ".nox", ".mypy_cache", ".pytest_cache",
            ".eggs", "*.egg-info", "dist", "build", ".venv", "venv",
            "env", ".env",
        }

    def should_ignore(self, path: str) -> bool:
        rel = os.path.relpath(path, self.root)
        parts = rel.split(os.sep)

        for part in parts:
            if part in self.always_ignore:
                return True
            for pattern in self.always_ignore:
                if pattern.startswith("*") and part.endswith(pattern[1:]):
                    return True

        for pattern in self.patterns:
            pattern = pattern.rstrip("/")
            if pattern in parts:
                return True
            # Simple glob match
            if pattern.startswith("*"):
                suffix = pattern[1:]
                if any(p.endswith(suffix) for p in parts):
                    return True
        return False


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def _extract_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    """Return (module_name, line_number) for all imports."""
    imports: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
    return imports


def _module_name_from_path(path: str, root: str) -> str:
    """Convert file path to Python module name."""
    rel = os.path.relpath(path, root)
    rel = rel.replace(os.sep, ".")
    if rel.endswith(".__init__.py"):
        rel = rel[:-12]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    return rel


# ---------------------------------------------------------------------------
# Circular import detection (Tarjan's SCC)
# ---------------------------------------------------------------------------

def _find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find all strongly connected components with > 1 node."""
    index_counter = [0]
    stack: List[str] = []
    lowlink: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Set[str] = set()
    sccs: List[List[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in graph:
        if v not in index:
            strongconnect(v)

    return sccs


# ---------------------------------------------------------------------------
# Python 2/3 compatibility checks
# ---------------------------------------------------------------------------

class _Py2Detector(ast.NodeVisitor):
    """Detect Python 2-specific constructs."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def visit_Print(self, node: ast.AST) -> None:
        self.issues.append("print statement (Python 2 only)")

    def visit_Raise(self, node: ast.Raise) -> None:
        # Python 2 style: raise Exception, "msg"
        if node.cause is None and node.exc is not None:
            if isinstance(node.exc, ast.Tuple):
                self.issues.append("Python 2-style raise with tuple")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id == "raw_input":
                self.issues.append(f"raw_input() is Python 2 only (line {node.lineno})")
            if node.func.id == "xrange":
                self.issues.append(f"xrange() is Python 2 only (line {node.lineno})")
            if node.func.id == "unicode":
                self.issues.append(f"unicode() is Python 2 only (line {node.lineno})")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "has_key":
                self.issues.append(f".has_key() is Python 2 only (line {node.lineno})")
            if node.func.attr == "iteritems":
                self.issues.append(f".iteritems() is Python 2 only (line {node.lineno})")
            if node.func.attr == "itervalues":
                self.issues.append(f".itervalues() is Python 2 only (line {node.lineno})")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Check for <> operator (handled by AST as NotEq, but with different syntax)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

def _compute_health_score(report: ProjectReport) -> float:
    """Compute a 0-100 health score based on various metrics."""
    score = 100.0

    if report.files_analyzed == 0:
        return score

    # Bug density penalty
    bug_density = report.total_bugs / max(report.files_analyzed, 1)
    score -= min(30, bug_density * 5)

    # Error severity penalty
    error_count = report.bugs_by_severity.get("ERROR", 0)
    score -= min(20, error_count * 2)

    # Circular import penalty
    score -= min(15, len(report.circular_imports) * 5)

    # Complexity penalty
    if report.complexity_scores:
        avg_cc = sum(report.complexity_scores.values()) / len(report.complexity_scores)
        if avg_cc > 10:
            score -= min(15, (avg_cc - 10) * 2)

    # Taint flow penalty
    score -= min(10, report.taint_flows * 3)

    # Python 2 compatibility penalty
    score -= min(10, len(report.py2_incompatibilities) * 1)

    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

class ProjectScanner:
    """Scan an entire Python project and aggregate analysis results."""

    def __init__(self) -> None:
        self.ast_analyzer_cls = PythonASTAnalyzer
        self.complexity_analyzer_cls = ComplexityAnalyzer
        self.pattern_detector_cls = PatternDetector
        self.taint_tracker_cls = TaintTracker

    def scan(self, project_root: str) -> ProjectReport:
        report = ProjectReport()
        root = os.path.abspath(project_root)

        if not os.path.isdir(root):
            report.recommendations.append(f"Path '{root}' is not a directory")
            return report

        gitignore = _GitignoreFilter(root)

        # Collect all .py files
        py_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter directories in-place
            dirnames[:] = [d for d in dirnames
                           if not gitignore.should_ignore(os.path.join(dirpath, d))]
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    full = os.path.join(dirpath, fname)
                    if not gitignore.should_ignore(full):
                        py_files.append(full)

        # Build module -> path mapping
        module_paths: Dict[str, str] = {}
        for path in py_files:
            mod = _module_name_from_path(path, root)
            module_paths[mod] = path

        # Dependency graph
        dep_graph: Dict[str, Set[str]] = defaultdict(set)

        # Analyse each file
        for path in py_files:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except (OSError, IOError):
                continue

            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError:
                report.total_bugs += 1
                report.bugs_by_severity["ERROR"] += 1
                continue

            rel_path = os.path.relpath(path, root)
            mod_name = _module_name_from_path(path, root)
            file_report = FileReport(path=rel_path, lines=len(source.split("\n")))

            # AST analysis
            analyzer = self.ast_analyzer_cls(filename=rel_path)
            result = analyzer.analyze_source(source)
            file_report.bugs = result.bugs

            # Complexity
            ca = self.complexity_analyzer_cls(filename=rel_path)
            cr = ca.analyze(source)
            file_report.complexity = cr

            # Pattern detection
            pd = self.pattern_detector_cls(filename=rel_path)
            file_report.patterns = pd.detect(source)

            # Taint analysis
            tt = self.taint_tracker_cls()
            file_report.taint_flows = tt.analyze(source, filename=rel_path)

            # Imports / dependencies
            imports = _extract_imports(tree)
            file_report.imports = [m for m, _ in imports]
            for imp_mod, imp_line in imports:
                # Check if it's an internal module
                parts = imp_mod.split(".")
                for i in range(len(parts), 0, -1):
                    candidate = ".".join(parts[:i])
                    if candidate in module_paths:
                        dep_graph[mod_name].add(candidate)
                        report.dependency_graph.append(DependencyEdge(
                            source=mod_name, target=candidate, import_line=imp_line
                        ))
                        break

            # Python 2 check
            py2 = _Py2Detector()
            py2.visit(tree)
            for issue in py2.issues:
                report.py2_incompatibilities.append(f"{rel_path}: {issue}")

            # Aggregate counts
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    report.total_functions += 1
                if isinstance(node, ast.ClassDef):
                    report.total_classes += 1

            # Store
            report.file_reports[rel_path] = file_report
            report.total_lines += file_report.lines

            # Aggregate bugs
            for bug in file_report.bugs:
                report.total_bugs += 1
                sev = bug.severity.name
                report.bugs_by_severity[sev] = report.bugs_by_severity.get(sev, 0) + 1
                report.bugs_by_category[bug.category] = report.bugs_by_category.get(bug.category, 0) + 1
                report.bugs_by_file[rel_path] = report.bugs_by_file.get(rel_path, 0) + 1

            # Aggregate complexity
            if cr.per_function:
                avg_cc = sum(f.cyclomatic for f in cr.per_function) / len(cr.per_function)
                report.complexity_scores[rel_path] = avg_cc

            report.taint_flows += len(file_report.taint_flows)

        report.files_analyzed = len(py_files)

        # Circular imports
        report.circular_imports = _find_cycles(dep_graph)

        # Health score
        report.health_score = _compute_health_score(report)

        # Recommendations
        if report.circular_imports:
            report.recommendations.append(
                f"Found {len(report.circular_imports)} circular import cycle(s)"
            )
        if report.bugs_by_severity.get("ERROR", 0) > 0:
            report.recommendations.append(
                f"Fix {report.bugs_by_severity['ERROR']} error-level bug(s)"
            )
        if report.taint_flows > 0:
            report.recommendations.append(
                f"Address {report.taint_flows} potential security vulnerability(ies)"
            )
        if report.py2_incompatibilities:
            report.recommendations.append(
                f"Remove {len(report.py2_incompatibilities)} Python 2 construct(s)"
            )
        if report.complexity_scores:
            worst = max(report.complexity_scores.items(), key=lambda x: x[1])
            if worst[1] > 10:
                report.recommendations.append(
                    f"Reduce complexity in '{worst[0]}' (avg cyclomatic: {worst[1]:.1f})"
                )

        return report
